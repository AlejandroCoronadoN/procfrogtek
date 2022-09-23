import pandas as pd
import numpy as np

from datetime import datetime
import pandas as pd 
import sys


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
        df_ticket = frogtek_upc_raw_prepared.get_dataframe()
    else:
       #  entregas_path = "../../data/processed/preproc/entregas_preproc_" + TIMESTAMP + ".csv"
        # df_bitacoras.to_csv(entregas_path, index =False )
        ticket_paths = ["../../data/raw/frogtek_ticket_raw_h1_2021.csv", "../../data/raw/frogtek_ticket_raw_h2_2021.csv",
                        "../../data/raw/frogtek_ticket_raw_h1_2022.csv", "../../data/raw/frogtek_ticket_raw_h2_2022.csv"]
        df_all=pd.DataFrame()
        total_length =0
        total_tickets =[]
        total_tiendas = []
        for ticket_path in ticket_paths:
            df_ticket = pd.read_csv(ticket_path )
            df_ticket.columns  
            rename_dict = {
                'Periodo':'periodo',
                'Plaza':'region',
                'POS':'id_tienda',
                'Ticket':'ticket_id',
                'Descripción UPC': 'sku',
                'Unidades':'unidades',
                'Precio Venta':'precio_venta'	,
                'Venta Total':'venta',	
                'Meses con ventas "Vuala Swich" 2021':'meses_con_ventas'
                }
            df_ticket = df_ticket.rename(columns=rename_dict)
            total_length += len(df_ticket)
            total_tickets.extend( df_ticket.ticket_id.unique())
            total_tiendas.extend( df_ticket.id_tienda.unique())

            tienda_ids = ["periodo", "region", "id_tienda", 'sku']
            df_tienda_mean = df_ticket.groupby(tienda_ids)["precio_venta"].mean().reset_index()
            df_tienda_sum = df_ticket.groupby(tienda_ids)["unidades"].sum().reset_index()
            df_tienda = df_tienda_sum.merge(df_tienda_mean, on =tienda_ids)
            
            df_tienda["year"] = df_tienda["periodo"].apply(lambda x: x.split('-')[0])
            df_tienda["month"] = df_tienda["periodo"].apply(lambda x: x.split('-')[1])
            if len(df_all) ==0:
                df_all =df_tienda
            else:
                df_all = df_all.append(df_tienda)
        df_all.to_csv("../../data/processed/tienda_preproc.py", index=False)

    