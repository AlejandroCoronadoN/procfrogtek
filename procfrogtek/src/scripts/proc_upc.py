df_cumsum = df_diff.replace(np.nan, 0).groupby([ "Plaza", "Descripción UPC", "year"]).cumsum()
cumsum_col = col + "_cumsum"
df_cumsum = df_magn.rename(columns= {diff_col:cumsum_col})
df_sorted[cumsum_col] = np.abs(df_cumsum[cumsum_col])


df_min = df_sorted[df_sorted[cumsum_col]!=0]
df_min = df_min.replace(np.nan, 0).groupby([ "Plaza", "Descripción UPC", "year"]).min()
min_col = col + "_min"
df_min = df_min.rename(columns= {cumsum_col:min_col}).reset_index()
df_min = df_min[["Plaza", "Descripción UPC", "year"] + [min_col]]
df_sorted = df_sorted.merge(df_min, on=["Plaza", "Descripción UPC", "year"] )

df_max = df_sorted[df_sorted[cumsum_col]!=0]
df_max = df_max.replace(np.nan, 0).groupby([ "Plaza", "Descripción UPC", "year"]).max()
min_col = col + "_max"
df_max = df_max.rename(columns= {cumsum_col:max_col}).reset_index()
df_max = df_max[["Plaza", "Descripción UPC", "year"] + [max_col]]
df_sorted = df_sorted.merge(df_max, on=["Plaza", "Descripción UPC", "year"] )

df_direction = df_sorted[df_sorted[cumsum_col]!=0]
df_direction = df_direction.replace(np.nan, 0).groupby([ "Plaza", "Descripción UPC", "year"])[cumsum_col].last().reset_index()
dir_col = col + "_direction"
df_direction[dir_col] =0 
df_direction.loc[df_direction[cumsum_col]>0, dir_col] = 1
df_direction.loc[df_direction[cumsum_col]<0, dir_col] = -1

df_direction = df_direction[["Plaza", "Descripción UPC", "year"] + [dir_col]]
df_sorted = df_sorted.merge(df_direction, on=["Plaza", "Descripción UPC", "year"] )

df_direction = df_sorted[df_sorted[cumsum_col]!=0]
df_direction = df_direction.replace(np.nan, 0).groupby([ "Plaza", "Descripción UPC", "year"])[cumsum_col].last().reset_index()
dir_col = col + "_direction"
df_direction[dir_col] =0 
df_direction.loc[df_direction[cumsum_col]>0, dir_col] = 1
df_direction.loc[df_direction[cumsum_col]<0, dir_col] = -1


df_mean = df_sorted.groupby([ "Plaza", "Descripción UPC", "year"])[col].mean().reset_index()
mean_col = col + "_mean"
df_mean=df_mean.rename(columns={col:mean_col})
df_sorted = df_sorted.merge(df_mean, on =["Plaza", "Descripción UPC", "year"])

df_sorted
