from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


# For product labels: produces the Time_Coordinates element.

time_coordinates = interpret_template("""<Time_Coordinates>
      <start_date_time><NODE name="start_date_time"/></start_date_time>
      <stop_date_time><NODE name="stop_date_time"/></stop_date_time>
    </Time_Coordinates>""")


class TimeCoordinatesLabelReduction(Reduction):
    def reduce_fits_file(self, file, get_reduced_hdus):
        res = get_reduced_hdus()[0]
        assert isinstance(res, dict)
        return res

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        if n == 0:
            try:
                date_obs = header_unit['DATE-OBS']
                time_obs = header_unit['TIME-OBS']
                exptime = header_unit['EXPTIME']
                start_date_time = '%sT%s' % (date_obs, time_obs)
                stop_date_time = julian.tai_from_iso(start_date_time) + exptime
                stop_date_time = remove_trailing_decimal(stop_date_time)
            except KeyError:
                # Insert placeholders
                start_date_time = '2000-01-02Z'
                stop_date_time = '2000-01-02Z'
            tc = time_coordinates({'start_date_time': start_date_time,
                                   'stop_date_time': stop_date_time})
            return {'Time_Coordinates': tc}
        else:
            pass
