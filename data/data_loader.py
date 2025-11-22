"""
Data Loader - Quantum Trader Pro
Chargement et gestion des données de marché historiques et en temps réel
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
import time

class DataLoader:
    """
    Gestionnaire de données avec:
    - Chargement données historiques
    - Cache intelligent
    - Validation données
    - Resampling multi-timeframe
    - Mise à jour temps réel
    """
    
    def __init__(self, client, config: Dict):
        """
        Initialise le data loader
        
        Args:
            client: Instance BinanceClient
            config: Configuration complète du bot
        """
        self.client = client
        self.config = config
        self.logger = setup_logger('DataLoader')

        # Configuration - accès sécurisé
        self.symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')
        timeframes_config = get_nested_config(config, 'timeframes', default={})
        self.trend_tf = timeframes_config.get('trend', '1h')
        self.signal_tf = timeframes_config.get('signal', '5m')
        self.micro_tf = timeframes_config.get('micro', '1m')

        lookback = timeframes_config.get('lookback', {})
        self.lookback_short = lookback.get('short', 100)
        self.lookback_medium = lookback.get('medium', 500)
        self.lookback_long = lookback.get('long', 1000)

        # Cache
        self.cache = {}
        performance_config = get_nested_config(config, 'performance', default={})
        self.cache_enabled = performance_config.get('cache_indicators', True)
        self.cache_ttl = performance_config.get('cache_ttl_seconds', 300)
        self.cache_timestamps = {}
        
        # État
        self.last_update = {}
        
        self.logger.info("✅ Data Loader initialisé")
    
    def load_historical_data(
        self,
        symbol: Optional[str] = None,
        timeframe: str = '1h',
        limit: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Charge les données historiques
        
        Args:
            symbol: Paire de trading (défaut: config)
            timeframe: Timeframe (1m, 5m, 1h, etc.)
            limit: Nombre de bougies
            start_date: Date de début (format: YYYY-MM-DD)
            end_date: Date de fin
        
        Returns:
            DataFrame avec colonnes: timestamp, open, high, low, close, volume
        """
        
        symbol = symbol or self.symbol
        
        self.logger.info(
            f"📥 Chargement données: {symbol} {timeframe} (limit: {limit})"
        )
        
        try:
            # Vérifier cache
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if self._is_cache_valid(cache_key):
                self.logger.debug(f"✅ Données récupérées du cache: {cache_key}")
                return self.cache[cache_key].copy()

            # Utiliser pagination si plus de 1000 bougies demandées
            if limit > 1000:
                # Utiliser fetch_historical avec pagination
                df = self._fetch_with_pagination(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=self._parse_date(start_date) if start_date else None
                )
            else:
                # Charger depuis exchange (méthode standard)
                ohlcv = self.client.fetch_ohlcv(
                    timeframe=timeframe,
                    limit=limit,
                    since=self._parse_date(start_date) if start_date else None
                )

                if ohlcv is None or len(ohlcv) == 0:
                    self.logger.error(f"❌ Aucune donnée reçue pour {symbol} {timeframe}")
                    return pd.DataFrame()

                # Convertir en DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )

                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

            # Valider données
            df = self._validate_data(df)
            
            # Mettre en cache
            if self.cache_enabled:
                self.cache[cache_key] = df.copy()
                self.cache_timestamps[cache_key] = time.time()
            
            self.logger.info(
                f"✅ {len(df)} bougies chargées pour {symbol} {timeframe}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            return pd.DataFrame()
    
    def load_multi_timeframe_data(
        self,
        symbol: Optional[str] = None,
        timeframes: Optional[List[str]] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Charge les données pour plusieurs timeframes
        
        Args:
            symbol: Paire de trading
            timeframes: Liste des timeframes
            limit: Nombre de bougies par TF
        
        Returns:
            Dict {timeframe: DataFrame}
        """
        
        symbol = symbol or self.symbol
        timeframes = timeframes or [self.trend_tf, self.signal_tf, self.micro_tf]
        
        self.logger.info(
            f"📥 Chargement multi-timeframe: {symbol} {timeframes}"
        )
        
        data = {}
        for tf in timeframes:
            df = self.load_historical_data(symbol, tf, limit)
            if not df.empty:
                data[tf] = df
        
        return data
    
    def update_data(
        self,
        existing_df: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: str = '1h',
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Met à jour un DataFrame existant avec nouvelles bougies
        
        Args:
            existing_df: DataFrame existant
            symbol: Paire de trading
            timeframe: Timeframe
            limit: Nombre de nouvelles bougies à charger
        
        Returns:
            DataFrame mis à jour
        """
        
        symbol = symbol or self.symbol
        
        try:
            # Charger nouvelles données
            new_ohlcv = self.client.fetch_ohlcv(
                timeframe=timeframe,
                limit=limit
            )
            
            if new_ohlcv is None or len(new_ohlcv) == 0:
                return existing_df
            
            # Convertir en DataFrame
            new_df = pd.DataFrame(
                new_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            new_df.set_index('timestamp', inplace=True)
            
            # Fusionner avec données existantes
            combined = pd.concat([existing_df, new_df])
            
            # Supprimer duplicatas (garder les derniers)
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Trier par timestamp
            combined = combined.sort_index()
            
            return combined
            
        except Exception as e:
            self.logger.error(f"❌ Erreur mise à jour données: {e}")
            return existing_df
    
    def get_latest_candle(
        self,
        symbol: Optional[str] = None,
        timeframe: str = '1m'
    ) -> Optional[Dict]:
        """
        Récupère la dernière bougie
        
        Args:
            symbol: Paire de trading
            timeframe: Timeframe
        
        Returns:
            Dict avec OHLCV ou None
        """
        
        symbol = symbol or self.symbol
        
        try:
            ohlcv = self.client.fetch_ohlcv(timeframe, limit=1)
            
            if ohlcv is None or len(ohlcv) == 0:
                return None
            
            candle = ohlcv[-1]
            return {
                'timestamp': pd.to_datetime(candle[0], unit='ms'),
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération dernière bougie: {e}")
            return None
    
    def resample_data(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample les données vers un autre timeframe
        
        Args:
            df: DataFrame source
            target_timeframe: Timeframe cible (ex: '1h', '4h')
        
        Returns:
            DataFrame resample
        """
        
        try:
            # Mapping timeframes pandas
            tf_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            pandas_tf = tf_map.get(target_timeframe)
            if not pandas_tf:
                self.logger.warning(f"⚠️ Timeframe non supporté: {target_timeframe}")
                return df
            
            # Resample
            resampled = df.resample(pandas_tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Supprimer NaN
            resampled = resampled.dropna()
            
            self.logger.debug(
                f"✅ Données resample vers {target_timeframe}: {len(resampled)} bougies"
            )
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"❌ Erreur resampling: {e}")
            return df
    
    def _fetch_with_pagination(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Récupère plus de 1000 bougies en utilisant la pagination

        Args:
            symbol: Paire de trading
            timeframe: Timeframe
            limit: Nombre total de bougies désirées
            since: Timestamp de début (ms)

        Returns:
            DataFrame avec les bougies
        """
        all_data = []
        batch_size = 1000
        remaining = limit
        current_since = since
        batch_count = 0
        consecutive_empty = 0

        self.logger.info(f"📥 Pagination: chargement de {limit} bougies...")
        if current_since:
            from datetime import datetime
            start_dt = datetime.fromtimestamp(current_since / 1000)
            self.logger.info(f"📅 Début depuis: {start_dt}")

        while remaining > 0:
            batch_limit = min(batch_size, remaining)
            batch_count += 1

            try:
                ohlcv = self.client.fetch_ohlcv(
                    timeframe=timeframe,
                    limit=batch_limit,
                    since=current_since
                )

                if ohlcv is None or len(ohlcv) == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        self.logger.warning(f"⚠️ 3 requêtes vides consécutives, fin de pagination")
                        break
                    # Petit délai avant de réessayer
                    import time
                    time.sleep(0.5)
                    continue

                consecutive_empty = 0
                all_data.extend(ohlcv)
                remaining -= len(ohlcv)

                # Mise à jour du curseur pour le prochain batch
                current_since = ohlcv[-1][0] + 1

                # Progress log tous les 5 batchs
                if batch_count % 5 == 0:
                    self.logger.info(f"   📊 Batch {batch_count}: {len(all_data)} bougies collectées...")

                # Si moins de bougies reçues que demandées = fin des données disponibles
                if len(ohlcv) < batch_limit:
                    self.logger.info(f"   ℹ️ Fin des données disponibles (reçu {len(ohlcv)} < {batch_limit} demandées)")
                    break

                # Petit délai pour éviter rate limiting
                import time
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"❌ Erreur pagination batch {batch_count}: {e}")
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                import time
                time.sleep(1)

        if not all_data:
            self.logger.error(f"❌ Aucune donnée reçue pour {symbol} {timeframe}")
            if since:
                self.logger.warning(f"   ⚠️ Le testnet Binance peut avoir des données historiques limitées")
                self.logger.warning(f"   💡 Essayez avec moins de jours ou utilisez Binance mainnet (lecture seule)")
            return pd.DataFrame()

        # Convertir en DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convertir timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
        df = df.sort_index()

        # Limiter au nombre exact demandé
        if len(df) > limit:
            df = df.tail(limit)

        self.logger.info(f"✅ Pagination terminée: {len(df)} bougies récupérées en {batch_count} batchs")

        # Avertissement si moins que demandé
        if len(df) < limit:
            self.logger.warning(f"   ⚠️ Seulement {len(df)}/{limit} bougies disponibles")
            self.logger.warning(f"   💡 Le testnet Binance a des données historiques limitées")

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valide et nettoie les données

        Args:
            df: DataFrame à valider

        Returns:
            DataFrame nettoyé
        """

        # Supprimer NaN
        df = df.dropna()
        
        # Vérifier valeurs négatives
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            self.logger.warning("⚠️ Valeurs négatives détectées, suppression...")
            df = df[(df[['open', 'high', 'low', 'close', 'volume']] >= 0).all(axis=1)]
        
        # Vérifier cohérence OHLC
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            num_invalid = invalid_rows.sum()
            self.logger.warning(
                f"⚠️ {num_invalid} bougies invalides supprimées"
            )
            df = df[~invalid_rows]
        
        # Vérifier duplicatas
        duplicates = df.index.duplicated(keep='last')
        if duplicates.any():
            num_duplicates = duplicates.sum()
            self.logger.warning(
                f"⚠️ {num_duplicates} timestamps dupliqués supprimés"
            )
            df = df[~duplicates]
        
        return df
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifie si le cache est encore valide"""
        
        if not self.cache_enabled:
            return False
        
        if cache_key not in self.cache:
            return False
        
        # Vérifier TTL
        cache_time = self.cache_timestamps.get(cache_key, 0)
        age = time.time() - cache_time
        
        return age < self.cache_ttl
    
    def _parse_date(self, date_str: str) -> int:
        """Convertit une date string en timestamp ms"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return int(dt.timestamp() * 1000)
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur parsing date {date_str}: {e}")
            return None
    
    def clear_cache(self):
        """Vide le cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("🗑️ Cache vidé")
    
    def get_cache_info(self) -> Dict:
        """Retourne des infos sur le cache"""
        return {
            'enabled': self.cache_enabled,
            'size': len(self.cache),
            'ttl_seconds': self.cache_ttl,
            'keys': list(self.cache.keys())
        }
    
    def warmup(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Précharge les données en cache (warmup)
        
        Args:
            symbols: Liste des symboles à précharger
        
        Returns:
            True si succès
        """
        
        symbols = symbols or [self.symbol]
        timeframes = [self.trend_tf, self.signal_tf, self.micro_tf]
        
        self.logger.info(f"🔥 Warmup cache: {len(symbols)} symboles × {len(timeframes)} TF")
        
        success = True
        for symbol in symbols:
            for tf in timeframes:
                df = self.load_historical_data(symbol, tf, self.lookback_medium)
                if df.empty:
                    success = False
                    self.logger.error(f"❌ Échec warmup: {symbol} {tf}")
        
        if success:
            self.logger.info("✅ Warmup terminé avec succès")
        
        return success
