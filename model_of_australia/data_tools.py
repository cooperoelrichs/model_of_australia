class DataTools():
    def deflate(data, cpi, base_date, date_range, fields):
        cpi_filter = ((cpi['date'] >= date_range[0]) & (cpi['date'] <= date_range[1]))
        dat_filter = ((data['date'] >= date_range[0]) & (data['date'] <= date_range[1]))

        base_cpi = cpi[cpi['date'] == base_date]['value'].values
        data.loc[dat_filter, fields[1]] = (
            data[dat_filter][fields[0]].values / cpi[cpi_filter]['value'].values * base_cpi
        )
        return data
