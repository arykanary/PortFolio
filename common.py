from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import warnings
from time import sleep
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sqlite3
import pandas as pd
from typing import List

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.facecolor'] = '#718D99'


class SqliteOperator:
    """Operator class for setting, getting & changing sqlite databases

    https://www.sqlite.org/datatype3.html
    """
    def __init__(self, db_path):
        self.connector = sqlite3.connect(db_path)
        self.db_path = db_path

    def get_all_tables(self):
        """Get all the available tables"""
        return pd.read_sql(f"SELECT * FROM sqlite_master", self.connector, index_col='name')

    def get_table(self, table_name: str, col: (str, List[str]) = '*', index_col: str = None):
        """Get the data in the specified table"""
        col = ','.join(col) if isinstance(col, list) else col
        return pd.read_sql_query(f"SELECT {col} FROM {table_name}", self.connector, index_col=index_col)

    def get_dtypes(self, table_name: str = None):
        """Get the dtypes of the columns in the specified table"""
        return pd.read_sql(f"PRAGMA TABLE_INFO({table_name});", self.connector, index_col='cid')

    def add_lines(self, table_name: str, insert_df: pd.DataFrame):
        """Add data to the specified table"""
        for ind, row in insert_df.iterrows():
            row = row.dropna()
            col = ','.join(row.index.values)
            val = "','".join(['%s' % x for x in row.values])
            self.connector.execute(f"INSERT INTO {table_name} ( {col} ) VALUES ( '{val}' );")
            self.connector.commit()


class RWSData:
    """
    https://rijkswaterstaat.github.io/wm-ws-dl/#introduction

    >>> rws = RWSData(
    >>>     loc=('ARNH', 700021.921999557, 5762290.374687570),
    >>>     compartment='OW',
    >>>     unit='cm',
    >>>     quantity="WATHTE",
    >>>     characteristic='NAP',
    >>>     group='Dag',
    >>>     fix_data=True
    >>> )
    >>> df = rws.data_loc_datetime(
    >>>     start=datetime(1999, 12, 1)-timedelta(days=7)
    >>>     end=datetime(1999, 12, 1)
    >>> )
    
    """
    main_path = 'https://waterwebservices.rijkswaterstaat.nl/'
    collect_catalogus = '%sMETADATASERVICES_DBO/OphalenCatalogus/' % main_path
    collect_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenWaarnemingen' % main_path
    collect_count_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/CheckWaarnemingenAanwezig' % main_path
    collect_latest_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenLaatsteWaarnemingen' % main_path
    collect_grouped_observations = '%sONLINEWAARNEMINGENSERVICES_DBO/OphalenAantalWaarnemingen' % main_path

    def __init__(self, loc=('ARNH', 700021.921999557, 5762290.374687570),
                 compartment='OW', unit='cm', quantity="WATHTE", characteristic='NAP', group='Dag',
                 fix_data=True):
        self.compartment = compartment
        self.unit = unit
        self.quantity = quantity
        self.characteristic = characteristic
        self.group = group
        self.fix_data = fix_data

        self.code, self.x, self.y = loc

    @staticmethod
    def data_to_str(start, end): return (f"{start.strftime('%Y-%m-%dT%H:%M:%S')}.000+00:00",
                                         f"{end.strftime('%Y-%m-%dT%H:%M:%S')}.000+00:00")

    def meta_descriptions(self):
        """

        :return:
        """
        spec_dict = {"Eenheden": True, "Grootheden": True, "Hoedanigheden": True, "Compartimenten": True, "Parameters": True}
        resp = requests.post(self.collect_catalogus, json={"CatalogusFilter": spec_dict})
        result = resp.json()

        if result['Succesvol']:
            df_meta_list = pd.DataFrame({k: flatten_sequence(v) for k, v in enumerate(result['AquoMetadataLijst'])}).T if 'AquoMetadataLijst' in result.keys() else None
            df_loc_list = pd.DataFrame(result['LocatieLijst']) if 'LocatieLijst' in result.keys() else None
            df_meta_loc_list = pd.DataFrame(result['AquoMetadataLocatieLijst']) if 'AquoMetadataLocatieLijst' in result.keys() else None
            df_height_list = pd.DataFrame(result['BemonsteringshoogteLijst']) if 'BemonsteringshoogteLijst' in result.keys() else None
            df_contractor_list = pd.DataFrame(result['OpdrachtgevendeInstantieLijst']) if 'OpdrachtgevendeInstantieLijst' in result.keys() else None
            df_quality_list = pd.DataFrame(result['KwaliteitswaardecodeLijst']) if 'KwaliteitswaardecodeLijst' in result.keys() else None
            df_ref_list = pd.DataFrame(result['ReferentievlakLijst']) if 'ReferentievlakLijst' in result.keys() else None
            df_status_list = pd.DataFrame(result['StatuswaardeLijst']) if 'LocaStatuswaardeLijsttieLijst' in result.keys() else None

            return (df_meta_list, df_loc_list, df_meta_loc_list, df_height_list, df_contractor_list,
                    df_quality_list, df_ref_list, df_status_list)

    def ddl_data(self):
        catalog_filter = {"Grootheden": True, "Parameters": True, "Compartimenten": True, "Hoedanigheden": True,
                          "Eenheden": True, "BemonsteringsApparaten": True, "BemonsteringsMethoden": True,
                          "BemonsteringsSoorten": True, "BioTaxon": True, "BioTaxon_Compartimenten": True,
                          "MeetApparaten": True, "MonsterBewerkingsMethoden": True, "Organen": True,
                          "PlaatsBepalingsApparaten": True, "Typeringen": True, "WaardeBepalingstechnieken": True,
                          "WaardeBepalingsmethoden": True, "WaardeBewerkingsmethoden": True}

        resp = requests.post(self.collect_catalogus, json={"CatalogusFilter": catalog_filter})
        result = resp.json()
        if result['Succesvol']:
            return (pd.DataFrame({k: flatten_sequence(v) for k, v in enumerate(result['AquoMetadataLijst'])}).T,
                    pd.DataFrame(result['LocatieLijst']),
                    pd.DataFrame(result['AquoMetadataLocatieLijst']),
                    pd.DataFrame(result['BemonsteringshoogteLijst']),
                    pd.DataFrame(result['OpdrachtgevendeInstantieLijst']),
                    pd.DataFrame(result['KwaliteitswaardecodeLijst']),
                    pd.DataFrame(result['ReferentievlakLijst']),
                    pd.DataFrame(result['StatuswaardeLijst']))

    def check_loc_datetime(self, start=datetime.now() - timedelta(days=4), end=datetime.now()):
        start, end = self.data_to_str(start, end)
        request = {
            "AquoMetadataLijst": [{
                "Compartiment": {"Code": self.compartment},
                "Grootheid": {"Code": self.quantity},
                "Eenheid": {"Code": self.unit}
            }],
            "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}],
            "Periode": {"Begindatumtijd": start, "Einddatumtijd": end}
        }
        resp = requests.post(self.collect_count_observations, json=request)
        true_false_rename = {False: False, True: True, 'False': False, 'True': True, 'false': False, 'true': True}
        return pd.Series(resp.json()).map(true_false_rename)

    def data_loc_datetime(self, start=datetime.now() - timedelta(days=4), end=datetime.now()):
        start, end = self.data_to_str(start, end)
        request = {"AquoPlusWaarnemingMetadata": {"AquoMetadata": {"Compartiment": {"Code": self.compartment},
                                                                   "Grootheid": {"Code": self.quantity},
                                                                   "Eenheid": {"Code": self.unit}}},
                   "Locatie": {"X": self.x, "Y": self.y, "Code": self.code},
                   "Periode": {"Begindatumtijd": start, "Einddatumtijd": end}}
        resp = requests.post(self.collect_observations, json=request)
        result = resp.json()
        return self.get_result(result)

    def data_loc_latest(self):
        request = {"AquoPlusWaarnemingMetadataLijst": [{"AquoMetadata": {"Compartiment": {"Code": self.compartment},
                                                                         "Grootheid": {"Code": self.quantity},
                                                                         "Eenheid": {"Code": self.unit}}}],
                   "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}]}

        resp = requests.post(self.collect_latest_observations, json=request)
        result = resp.json()
        return self.get_result(result)

    def data_loc_group(self):
        """This has so far no useful purpose"""
        request = {"AquoMetadataLijst": [{"Compartiment": {"Code": self.compartment},
                                          "Grootheid": {"Code": self.quantity},
                                          "Eenheid": {"Code": self.unit}}],
                   "Groeperingsperiode": self.group,
                   "LocatieLijst": [{"X": self.x, "Y": self.y, "Code": self.code}],
                   "Periode": {"Begindatumtijd": self.start, "Einddatumtijd": self.end}}

        resp = requests.post(self.collect_grouped_observations, json=request)
        result = resp.json()
        if result['Succesvol']:
            result_lst = []
            for observation in result['AantalWaarnemingenPerPeriodeLijst']:
                result_lst.append([pd.Series(observation['Locatie']),
                                   pd.json_normalize(observation['AantalMetingenPerPeriodeLijst']),
                                   pd.json_normalize(observation['AquoMetadata']).iloc[0]])
            return result_lst
        else:
            warnings.warn(result['Foutmelding'])

    def get_result(self, result):
        if result['Succesvol']:
            result_lst = []
            for observation in result['WaarnemingenLijst']:
                loca = pd.Series(observation['Locatie'])

                meas = pd.json_normalize(observation['MetingenLijst'])
                meas.iloc[:] = [[c[0] if isinstance(c, list) and len(c) == 1 else c for c in r] for r in meas.values]
                meas.set_index('Tijdstip', inplace=True)
                meas.sort_index(inplace=True)
                meas.index = meas.index.str.slice(0, -10).str.replace('T', ' ')
                meas.index = pd.to_datetime(meas.index, format='%Y-%m-%d %H:%M:%S')
                if self.fix_data:
                    meas = meas.loc[meas['WaarnemingMetadata.KwaliteitswaardecodeLijst'] != '99']
                    meas['Meetwaarde.Waarde_Numeriek'] = self.fix_spikes(meas['Meetwaarde.Waarde_Numeriek'])

                meta = pd.json_normalize(observation['AquoMetadata']).iloc[0]
                result_lst.append([loca, meas, meta])
            return result_lst
        else:
            warnings.warn(result['Foutmelding'])

    @staticmethod
    def fix_spikes(data: pd.Series):
        def catch_spikes(arr: np.ndarray):
            std = np.nanstd(arr)
            d = np.ones(arr.shape, dtype=bool)
            d[:-1] *= np.diff(arr) > 3 * std

            out = np.ones(arr.shape)
            out[d] = np.nan
            out[~d] = 1.
            return out

        def interp_nan(arr: np.ndarray):
            y = np.asarray(list(arr))
            nans, x = np.isnan(y), lambda z: z.nonzero()[0]
            if len(nans) <=1:
                return y
            elif sum(nans) == 0:
                return y
            else:
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
                return y

        new_data = data.copy()
        new_data.loc[:] = interp_nan(data.values * catch_spikes(data.values))
        return new_data

    def get_date_range(self, start: datetime, end: datetime, step=timedelta(days=180), groupby='days'):
        groupby_dict = {
            'hours'  : lambda df: df.groupby([df.index.year, df.index.month, df.index.day, df.index.hour]).mean(),
            'days'   : lambda df: df.groupby([df.index.year, df.index.month, df.index.day]).mean(),
            'months' : lambda df: df.groupby([df.index.yeardf.index.month]).mean(),
            'years'  : lambda df: df.groupby([df.index.year]).mean(),
        }
        dframes = []
        for n in range(0, int(np.ceil((end - start) / step))):
            try:
                print(n, start+step*n, start+step*(n+1))
                sub = self.check_loc_datetime(start+step*n, start+step*(n+1))
                if sub['Succesvol'] and sub['WaarnemingenAanwezig']:
                    out = self.data_loc_datetime(start+step*n, start+step*(n+1))[0][1]['Meetwaarde.Waarde_Numeriek']
                    out = groupby_dict[groupby](out)
                    dframes.append(out)
                    print('DF appended', out.shape)

                sleep(2)
            except Exception as e:
                print(f'{e.__class__.__name__} at line {e.__traceback__.tb_lineno} with message {e}')

        df = pd.concat(dframes)
        df = df.loc[df.index.drop_duplicates()]
        return df


def correlation_significance(data: pd.DataFrame):
    corr = pd.DataFrame(
        index=data.columns,
        columns=pd.MultiIndex.from_product([data.columns, ['Correlation', 'P-Value']])
    )
    for cc in data.columns:
        for cr in data.columns:
            p = pearsonr(data[cc], data[cr])
            corr.loc[cr, (cc, 'Correlation')] = p.statistic
            corr.loc[cr, (cc, 'P-Value')] = p.pvalue
    return corr


def get_pie_chart(ax: plt.axis, data: pd.DataFrame, name):
    grp = data.groupby(name).count().iloc[:, 0]
    min_cnt = max([20, grp.nlargest(10).min()])
    misc_brands = grp[(grp<=min_cnt)].index
    grp['Misc'] = grp.loc[misc_brands].sum()
    grp.drop(misc_brands, inplace=True)
    grp.name = f'{name.replace("_", " ").title()} Count'
    grp.plot.pie(
        ax=ax,
        ylabel='',
        title=grp.name
    )


def get_hist_chart(ax: plt.axis, data: pd.DataFrame, name):
    data[name].plot.hist(
        ax=ax,
        bins=int(20*np.unique(data[name]).size**.25),
        density=True,
        title=name.replace("_", " ").title(),
        histtype='step'
    )
    data[name].plot.kde(
        ax=ax,
        title=name.replace("_", " ").title(),
    )
    ax.set_xlim(data[name].min(), data[name].max())
    ax.legend(['Histogram', 'Kernel Density'])
