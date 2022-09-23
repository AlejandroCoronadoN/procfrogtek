import pandas as pd
import numpy as np
from tqdm import tqdm 
from datetime import datetime
import  statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd 
import sys

def ols_model():
    X = df_sku[['month', 'year', 'region', 'precio_venta', 'effect_price_increment', 'magnitude']]
    Y = df_sku['unidades']
    X = sm.add_constant(X)
    fit = sm.OLS(Y, X).fit()
    df_sku = pd.get_dummies(df_sku, columns=['month'])
    df_sku = pd.get_dummies(df_sku, columns=['year'])


def rename_multiindex(df:pd.DataFrame,  id_cols:list, excluded_cols:list =[]) -> pd.DataFrame:
    """Converts multindex after pivot table

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    all_new_cols = []
    for mcol in df.columns:
        if mcol[0] not in excluded_cols:
            if mcol[0] in id_cols:
                all_new_cols.append(mcol[0])
            else:
                new_col = mcol[0] +"_"  + mcol[1]
                all_new_cols.append(new_col)
    initial_cols = [y[0] for y in df.columns]
    df.columns = all_new_cols
    return df 

# df_test =rename_multiindex(df=df_tienda_unidades,  id_cols=id_cols)


def impute_values(df:pd.DataFrame):
    # processed_cols = []
    quota = len(df)*.4
    deleted_cols = []
    for col in tqdm(df.columns, " 🦃 Imputation:"):
        zero_condition = df[col] ==0
        zero_count = np.sum(zero_condition)
        if zero_count > quota:
            deleted_cols.append(col)
            # print(f"\n\nDELETED COL: {col}  -  zero_count: {zero_count}  -  quota: {quota}")
        else:
            impute_value = df.loc[~zero_condition, col].mean()
            df.loc[zero_condition, col] = impute_value
            # unique_vals = df[col].unique()
            # print(f"\n\ncol: {col} \n\tvals: {unique_vals} \n\tzero_count:{zero_count}")
            # processed_cols.append(col)
    df = df[[x for x in df.columns if x not in deleted_cols]]
    return df 


def trime_cols(df1:pd.DataFrame, df2:pd.DataFrame):
    """
    """
    del_df1 = [x for x in df1.columns if x not in df2.columns]
    del_df2 = [x for x in df2.columns if x not in df1.columns]
    count_1 = 0 
    count_2 = 0 

    for col in del_df1:
        del df1[col]
        count_1+=1
    for col in del_df2:
        del df2[col]
        count_2+=1
    print(f"\n\t * Deleted columns from df1: {count_1} \n\t * Deleted_columns from df2: {count_2}")
    return df1, df2


def impute_var_change(df):
    processed_cols = []
    for col in df.columns:
        df[col]


def melt_pivoted(df:pd.DataFrame, col:str):
    df_melt = df.reset_index().melt(id_vars=id_cols)
    df_melt = df_melt.rename(columns={"value":col})
    return df_melt


def create_experiment(col:str, df:pd.DataFrame,  id_sort:list =[ 'region', 'sku', "year", "month"],  id_rolling:list = [ 'region', 'sku', "year"]) -> pd.DataFrame:
    """Returns a Data Frame with all the required columns to calcula differential experiment variables
    the main columns required to perform the experiments are
    
    maginitude: The maximim change in price (accumulated)/Average price '
        of SKU at id_rolling level
    effect_strategy: Identifies when the major price change was made and lavels 
        all months aftrer the effect with a falf 1 variable

    Args:
        col (str): _description_
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_sorted = df.sort_values(id_sort)
    df_month = df_sorted["month"]
    df_diff = df_sorted.groupby(id_rolling)[col].rolling(window=2).apply(my_fun).reset_index()
    diff_col = col + "_diff"
    df_diff = df_diff.rename(columns= {col:diff_col})
    df_sorted[diff_col] = df_diff[diff_col]

    df_cumsum = df_diff.replace(np.nan, 0).groupby(id_rolling).cumsum()
    cumsum_col = col + "_cumsum"
    df_cumsum = df_cumsum.rename(columns= {diff_col:cumsum_col})
    df_sorted[cumsum_col] = np.abs(df_cumsum[cumsum_col])
    
    #? REQUIRED FORM IMPACT CALC
    df_min_cumsum = df_sorted[df_sorted[cumsum_col]!=0]
    df_min_cumsum = df_min_cumsum.replace(np.nan, 0).groupby(id_rolling).min()
    cumsum_min_col = col + "_cumsum_min"
    df_min_cumsum = df_min_cumsum.rename(columns= {cumsum_col:cumsum_min_col}).reset_index()
    df_min_cumsum = df_min_cumsum[id_rolling+ [cumsum_min_col]]
    df_sorted = df_sorted.merge(df_min_cumsum, on=id_rolling)
    
    df_max_cumsum = df_sorted[df_sorted[cumsum_col]!=0]
    df_max_cumsum = df_max_cumsum.replace(np.nan, 0).groupby(id_rolling).max()
    cumsum_max_col = col + "_cumsum_max"
    df_max_cumsum = df_max_cumsum.rename(columns= {cumsum_col:cumsum_max_col}).reset_index()
    df_max_cumsum = df_max_cumsum[id_rolling+ [cumsum_max_col]]
    df_sorted = df_sorted.merge(df_max_cumsum, on=id_rolling)

    #? NON_REQUIRED
    df_min = df_sorted[df_sorted[cumsum_col]!=0]
    df_min = df_min.replace(np.nan, 0).groupby(id_rolling).min()
    min_col = col + "_min"
    df_min = df_min.rename(columns= {cumsum_col:min_col}).reset_index()
    df_min = df_min[id_rolling+ [min_col]]
    df_sorted = df_sorted.merge(df_min, on=id_rolling)
    
    df_max = df_sorted[df_sorted[cumsum_col]!=0]
    df_max = df_max.replace(np.nan, 0).groupby(id_rolling).max()
    max_col = col + "_max"
    df_max = df_max.rename(columns= {col:max_col}).reset_index()
    df_max = df_max[id_rolling+ [max_col]]
    df_sorted = df_sorted.merge(df_max, on=id_rolling)

    #? REQUIRED FOR EFFECT -MAGNITUD
    df_min_diff = df_sorted[df_sorted[diff_col]!=0]
    df_min_diff = df_min_diff.replace(np.nan, 0).groupby(id_rolling).min()
    diff_min_col = col + "_diff_min"
    df_min_diff = df_min_diff.rename(columns= {diff_col:diff_min_col}).reset_index()
    df_min_diff = df_min_diff[id_rolling+ [diff_min_col]]
    df_sorted = df_sorted.merge(df_min_diff, on=id_rolling)
    
    df_max_diff = df_sorted[df_sorted[diff_col]!=0]
    df_max_diff = df_max_diff.replace(np.nan, 0).groupby(id_rolling).max()
    diff_max_col = col + "_diff_max"
    df_max_diff = df_max_diff.rename(columns= {diff_col:diff_max_col}).reset_index()
    df_max_diff = df_max_diff[id_rolling+ [diff_max_col]]
    df_sorted = df_sorted.merge(df_max_diff, on=id_rolling)

    #! DIRECTION 
    df_direction = df_sorted[df_sorted[col]!=0]
    df_direction = df_direction.replace(np.nan, 0).groupby(id_rolling)[cumsum_col].last().reset_index()
    dir_col = col + "_direction"
    df_direction[dir_col] =0 
    df_direction.loc[df_direction[cumsum_col]>0, dir_col] = 1
    df_direction.loc[df_direction[cumsum_col]<0, dir_col] = -1

    df_direction = df_direction[id_rolling+ [dir_col]]
    df_sorted = df_sorted.merge(df_direction, on=id_rolling)
    df_mean = df_sorted.groupby(id_rolling)[col].mean().reset_index()
    mean_col = col + "_mean"
    df_mean=df_mean.rename(columns={col:mean_col})
    df_sorted = df_sorted.merge(df_mean, on =id_rolling)
    
    #! MAGINITUDE
    positive_direction_bool = df_sorted[dir_col]>0
    df_sorted.loc[positive_direction_bool,"magnitude"]  = df_sorted[cumsum_max_col]/ df_sorted[mean_col]
    df_sorted.loc[~positive_direction_bool,"magnitude"]  = df_sorted[cumsum_min_col]/ df_sorted[mean_col]
    df_sorted.loc[df_sorted[dir_col]==0,"magnitude"]  = 0
    df_sorted["magnitude"] = np.abs(df_sorted["magnitude"])
    
    #! EXPERIMENT EFFECT
    max_pricechange_bool = df_sorted[diff_max_col] == df_sorted[diff_col]
    min_pricechange_bool = df_sorted[diff_min_col] == df_sorted[diff_col]
    df_sorted[dir_col]>0 
    df_sorted["effect"] =0
    df_sorted.loc[(positive_direction_bool&max_pricechange_bool), "effect"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted.loc[(~positive_direction_bool&min_pricechange_bool), "effect"] =1#De los negativos cuando sucede el mayor caida de precio

    df_sorted = df_sorted.sort_values(id_sort)

    df_sorted["effect_strategy"] =np.nan
    df_sorted.loc[(positive_direction_bool&max_pricechange_bool), "effect_strategy"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted.loc[(~positive_direction_bool&min_pricechange_bool), "effect_strategy"] =-1#De los negativos cuando sucede el mayor caida de precio
    df_sorted["effect_strategy"] = df_sorted.groupby(id_rolling)["effect_strategy"].fillna(method='ffill').fillna(0)
    #Se etiqueta como 1 el perido a partir del cual se produce el mayor impacto de la eqtiqueta. Dependiendo de DIRECTION de la 
    # estrategia tendremos que mapear el valor para identificar AUMENTOS DE PRECIO VS ESTRATEGIA = AUMENTOS + CAIDAS

    df_sorted = df_sorted.sort_values(id_sort)

    df_sorted["effect_price_increment"] =np.nan
    df_sorted.loc[max_pricechange_bool, "effect_price_increment"] =1 #De los positivos cuando sucede el mayor aumento de precio
    df_sorted["effect_price_increment"] = df_sorted.groupby(id_rolling)["effect_price_increment"].fillna(method='ffill').fillna(0)


    return df_sorted

#########################################################################
#########################################################################
#########################################################################

if __name__ == "__main__":
    #TODO:  rename df_bitacoras_preproc and use it as df_bitacoras (until we get all client and sucursal data)
    DATAIKU = False
    PROCESS_SEMANA = False
    TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M") # Useful for saving files by date-time name.

    if DATAIKU:
        frogtek_upc_raw_prepared = dataiku.Dataset("frogtek_ticket_raw_prepared")
        df_tienda = frogtek_upc_raw_prepared.get_dataframe()
    else:
       #  entregas_path = "../../data/processed/preproc/entregas_preproc_" + TIMESTAMP + ".csv"
        # df_bitacoras.to_csv(entregas_path, index =False )
        tienda_path = "../../data/processed/tienda_preproc.py"
        df_tienda = pd.read_csv(tienda_path )
        df_tienda  = df_tienda[df_tienda.region != "Total"]
        #df_tienda = df_tienda.sample(100000)
        FILTER_YEAR = 2022
        if FILTER_YEAR:
            df_tienda = df_tienda[df_tienda.year == FILTER_YEAR]

    id_zona_cols = ['region', 'sku', 'month', 'year']

    df_tienda_group_zona = df_tienda.groupby(id_zona_cols)[['precio_venta', 'unidades']].mean().reset_index()
    df_tienda_group_zona = df_tienda_group_zona.rename(columns={'precio_venta':'pv_impute', 'unidades':'u_impute'}) 
    df_tienda = df_tienda.merge(df_tienda_group_zona, on =id_zona_cols, how = "left")
    
    df_tienda.loc[df_tienda.unidades.isna(), "unidades"] = df_tienda["u_impute"]
    df_tienda.loc[df_tienda.unidades.isna(), "precio_venta"] = df_tienda["pv_impute"]
    
    id_cols = ['region', 'id_tienda', 'sku', 'month', 'year']
    df_tienda_group = df_tienda.groupby(id_cols)[['precio_venta', 'unidades']].mean().reset_index()
    id_cols=['month', 'year']
    df_tienda_precioventa =df_tienda_group.pivot_table(index=id_cols, columns=['region', 'sku', 'id_tienda'],values= 'precio_venta', aggfunc=np.mean, fill_value=0)
    df_tienda_unidades =df_tienda_group.pivot_table(index=id_cols, columns=['region', 'sku', 'id_tienda'],values='unidades', aggfunc=np.mean, fill_value=0)

    #! IMPUTE VALUES
    df_tienda_precioventa_impute = impute_values(df_tienda_precioventa)
    df_tienda_unidades_impute = impute_values(df_tienda_unidades)
    
    #! TRIME COLUMNS
    df_tienda_precioventa_impute, df_tienda_unidades_impute = trime_cols(df_tienda_precioventa_impute, df_tienda_unidades_impute)

    #! MELTED
    df_tienda_unidades_melt = melt_pivoted(df_tienda_unidades_impute, "unidades")
    df_tienda_precioventa_melt =melt_pivoted(df_tienda_precioventa_impute, "precio_venta")
    
    def my_fun(x):
        return x.iloc[1]-x.iloc[0]
    
    # df_tienda_precioventa_melt.set_index(id_sort).sort_index().rolling(window=2).apply(my_fun)
    
    def prepare_quantity(df_unidades:pd.DataFrame, group_cols:list, col:str):
        df_unidades_avg= df_unidades.groupby(group_cols)["unidades"].mean().reset_index()
        df_unidades_avg = df_unidades_avg.rename(columns={"unidades":"unidades_avg"})

        df_unidades_std= df_unidades.groupby(group_cols)["unidades"].std().reset_index()
        df_unidades_std = df_unidades_std.rename(columns={"unidades":"unidades_std"})
        
        df_unidades =df_unidades.merge(df_unidades_avg, how="left", on =group_cols )
        df_unidades =df_unidades.merge(df_unidades_std, how="left", on =group_cols)
        df_unidades[col] = (df_unidades["unidades"] - df_unidades["unidades_avg"])/df_unidades["unidades_std"]
        return df_unidades
    
    df_exp_pv = create_experiment("precio_venta", df_tienda_precioventa_melt, id_sort =[ 'region', 'id_tienda', 'sku', "year", "month"],  id_rolling = [ 'region', 'id_tienda', 'sku', "year"])
    
    brand_dictionary = {
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'bimbo',
       'Kinder Delice Chocolate Bolsa 39 g': 'kinder',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'marinela',
       'Marinela Gansito Chocolate Bolsa 50 g': 'marinela',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'marinela',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'marinela',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'vuala',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'vuala',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'vuala',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala',
    }

    swich_dictionary = {
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'competencia_indirecta',
       'Kinder Delice Chocolate Bolsa 39 g': 'competencia_indirecta',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'competencia_indirecta',
       'Marinela Gansito Chocolate Bolsa 50 g': 'competencia_indirecta',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'competencia_indirecta',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'competencia_indirecta',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'vuala_other',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'vuala_other',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'vuala_other',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
    }

    swich_competencia_dictionary = {
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'competencia_indirecta',
       'Kinder Delice Chocolate Bolsa 39 g': 'competencia_indirecta',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'competencia_indirecta',
       'Marinela Gansito Chocolate Bolsa 50 g': 'competencia_indirecta',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'competencia_directa',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'competencia_directa',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'competencia_indirecta',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'competencia_indirecta',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'competencia_indirecta',
       'Vuala Swich Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'vuala_swich',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'vuala_swich',
    }
    
    swich_types_dictionary = { #Canibalizacion
       'Bimbo Crossantines Chocolate Bolsa 32 g': 'other',
       'Kinder Delice Chocolate Bolsa 39 g': 'other',
       'Marinela Choco Roles Mini PiÃ±a Bolsa 28 g': 'other',
       'Marinela Gansito Chocolate Bolsa 50 g': 'other',
       'Marinela Gansito Mini Chocolate Bolsa 24 g': 'other',
       'Marinela Pinguinos Mini Chocolate Bolsa 25 g': 'other',
       'Vuala Sorpresa Cajeta Bolsa 60 g': 'other',
       'Vuala Sorpresa Chocolate Bolsa 60 g': 'other',
       'Vuala Sorpresa Vainilla Bolsa 60 g': 'other',
       'Vuala Swich Chocolate Bolsa 32 g': 'swich',
       'Vuala Swich Pastelito Chocolate Bolsa 32 g': 'swich',
       'Vuala Swich Roll Chocolate Bolsa 32 g': 'roll',
       'Vuala Swich Roll Pastelito Chocolate Bolsa 32 g': 'roll',
    }
        
        
    df_tienda_unidades_melt["sku_brand"] = df_tienda_unidades_melt['sku'].map(brand_dictionary)
    df_tienda_unidades_melt["sku_swich"] = df_tienda_unidades_melt['sku'].map(swich_dictionary)
    df_tienda_unidades_melt["sku_competencia"] = df_tienda_unidades_melt['sku'].map(swich_competencia_dictionary)
    df_tienda_unidades_melt["sku_swich_types"] = df_tienda_unidades_melt['sku'].map(swich_types_dictionary)

    id_cols = [ 'region', 'id_tienda', 'sku', "year", "month"]
    df = df_exp_pv.merge(df_tienda_unidades_melt[id_cols + ["sku_brand", "sku_swich", 'sku_competencia']], on =id_cols)
    df = df.merge(df_tienda_unidades_melt[id_cols + ["unidades"]], on =id_cols)
    df_unidades = prepare_quantity(df_tienda_unidades_melt, group_cols = ['sku'], col = "unidades_sku_std")
    df = df.merge(df_unidades[id_cols + ["unidades_sku_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_tienda_unidades_melt, group_cols = ["sku_brand" ], col = "unidades_brand_std")
    df = df.merge(df_unidades[id_cols + ["unidades_brand_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_tienda_unidades_melt, group_cols = ["sku_swich" ], col = "unidades_swich_std")
    df = df.merge(df_unidades[id_cols + ["unidades_swich_std"]], on =id_cols)

    df_unidades = prepare_quantity(df_tienda_unidades_melt, group_cols = ["sku_competencia" ], col = "unidades_competencia_std")
    df = df.merge(df_unidades[id_cols + ["unidades_competencia_std"]], on =id_cols)
    
    df["unidades_std"] =  (df["unidades"] - df["unidades"].mean())/df["unidades"].std()
    df = df.dropna()

    df_ols_results = pd.DataFrame()
    i =0
    for sku in tqdm(df.sku.unique()):
        df_sku = df[df.sku == sku]
        model = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        fit = ols(model, data=df_sku).fit() 
        print(f"\n\n\n\n{'*'*120}\n{'*'*30}  {sku} {FILTER_YEAR} {'*'*30}\n{'*'*120}\n")
        print(fit.summary())
        
        coeff_precio_venta = fit.params["precio_venta"]
        coeff_price_increment = fit.params["C(effect_price_increment)[T.1.0]"]
        mean_unidades = df_sku.unidades.mean()
        elasticity = coeff_precio_venta/mean_unidades
        impact_price_increment = coeff_price_increment/mean_unidades
        df_ols_results.loc[i, "model"] = "Modelo Robusto"
        if FILTER_YEAR:
                df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month)'
        else:
            df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        df_ols_results.loc[i, "pval_precio_venta"] = fit.pvalues["precio_venta"]
        df_ols_results.loc[i, "pval_price_increment"] = fit.pvalues["C(effect_price_increment)[T.1.0]"]
        df_ols_results.loc[i, "coeff_precio_venta"] =coeff_precio_venta
        df_ols_results.loc[i, "coeff_incremento_precio"] =coeff_price_increment
        df_ols_results.loc[i, "mean_unidades"] = mean_unidades
        df_ols_results.loc[i, "elasticity"] = elasticity
        df_ols_results.loc[i, "impact_price_increment"] = impact_price_increment
        df_ols_results.loc[i, "subset"] = sku
        i+=1

    for brand in tqdm(df.sku_brand.unique()):
        df_brand = df[df.sku_brand == brand]
        model = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        fit = ols(model, data=df_brand).fit() 
        print(f"\n\n\n\n{'*'*120}\n{'*'*30}  {brand} {FILTER_YEAR} {'*'*30}\n{'*'*120}\n")
        print(fit.summary())
        
        coeff_precio_venta = fit.params["precio_venta"]
        coeff_price_increment = fit.params["C(effect_price_increment)[T.1.0]"]

        mean_unidades = df_brand.unidades.mean()
        elasticity = coeff_precio_venta/mean_unidades
        impact_price_increment = coeff_price_increment/mean_unidades

        df_ols_results.loc[i, "model"] = "Modelo Robusto"
        if FILTER_YEAR:
                df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month)'
        else:
            df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        df_ols_results.loc[i, "pval_precio_venta"] = fit.pvalues["precio_venta"]
        df_ols_results.loc[i, "pval_price_increment"] = fit.pvalues["C(effect_price_increment)[T.1.0]"]
        df_ols_results.loc[i, "coeff_precio_venta"] =coeff_precio_venta
        df_ols_results.loc[i, "coeff_incremento_precio"] =coeff_price_increment

        df_ols_results.loc[i, "mean_unidades"] = mean_unidades
        df_ols_results.loc[i, "elasticity"] = elasticity
        df_ols_results.loc[i, "impact_price_increment"] = impact_price_increment
        df_ols_results.loc[i, "subset"] = brand
        i+=1

    for competencia_group in tqdm(df.sku_competencia.unique()):
        df_competencia = df[df.sku_competencia == competencia_group]
        model = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        fit = ols(model, data=df_competencia).fit() 
        print(f"\n\n\n\n{'*'*120}\n{'*'*30}  {competencia_group} {FILTER_YEAR} {'*'*30}\n{'*'*120}\n")
        print(fit.summary())
        
        coeff_precio_venta = fit.params["precio_venta"]
        coeff_price_increment = fit.params["C(effect_price_increment)[T.1.0]"]

        mean_unidades = df_competencia.unidades.mean()
        elasticity = coeff_precio_venta/mean_unidades
        impact_price_increment = coeff_price_increment/mean_unidades
        df_ols_results.loc[i, "model"] = "Modelo Robusto"
        if FILTER_YEAR:
                df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month)'
        else:
            df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
        df_ols_results.loc[i, "pval_precio_venta"] = fit.pvalues["precio_venta"]
        df_ols_results.loc[i, "pval_price_increment"] = fit.pvalues["C(effect_price_increment)[T.1.0]"]
        df_ols_results.loc[i, "coeff_precio_venta"] =coeff_precio_venta
        df_ols_results.loc[i, "coeff_incremento_precio"] =coeff_price_increment
        df_ols_results.loc[i, "mean_unidades"] = mean_unidades
        df_ols_results.loc[i, "elasticity"] = elasticity
        df_ols_results.loc[i, "impact_price_increment"] = impact_price_increment
        df_ols_results.loc[i, "subset"] = competencia_group
        i+=1

    df_competencia = df[df.sku_swich== "vuala_other"] #Resto de productos swich
    model = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
    fit = ols(model, data=df_competencia).fit() 
    #print(f"\n\n\n\n{'*'*120}\n{'*'*30}  {competencia_group}  {'*'*30}\n{'*'*120}\n")
    #print(fit.summary())
    
    coeff_precio_venta = fit.params["precio_venta"]
    coeff_price_increment = fit.params["C(effect_price_increment)[T.1.0]"]
    mean_unidades = df_competencia.unidades.mean()
    elasticity = coeff_precio_venta/mean_unidades
    impact_price_increment = coeff_price_increment/mean_unidades
    df_ols_results.loc[i, "model"] = "Modelo Robusto"
    if FILTER_YEAR:
        df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month)'
    else:
        df_ols_results.loc[i, "modelo_str"] = 'unidades ~ precio_venta +magnitude + C(effect_price_increment) +  C(region) + C(month) + C(year)'
    df_ols_results.loc[i, "pval_precio_venta"] = fit.pvalues["precio_venta"]
    df_ols_results.loc[i, "pval_price_increment"] = fit.pvalues["C(effect_price_increment)[T.1.0]"]

    df_ols_results.loc[i, "coeff_precio_venta"] =coeff_precio_venta
    df_ols_results.loc[i, "coeff_incremento_precio"] =coeff_price_increment
    df_ols_results.loc[i, "mean_unidades"] = mean_unidades
    df_ols_results.loc[i, "elasticity"] = elasticity
    df_ols_results.loc[i, "impact_price_increment"] = impact_price_increment
    df_ols_results.loc[i, "subset"] = competencia_group
    i+=1

if FILTER_YEAR:
    df.to_csv(f"../../data/processed/tienda_proc_{FILTER_YEAR}.csv")
    df_ols_results.to_csv(f"../../data/processed/ols_results_{FILTER_YEAR}.csv")
else:
    df.to_csv(f"../../data/processed/tienda_proc.csv")
    df_ols_results.to_csv(f"../../data/processed/ols_results.csv")