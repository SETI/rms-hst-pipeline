from pdart.reductions.Reduction import *
from pdart.xml.Templates import *
from pdart.pds4labels.TargetIdentification import *


# For product labels: produces the Target_Identification element.

class TargetIdentificationLabelReduction(Reduction):
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
                targname = header_unit['TARGNAME']
                target = targname_to_target(targname)
            except KeyError:
                target = None

            if target is None:
                # Insert placeholder
                target_name = 'Magrathea'
                target_type = 'Planet'
                target_description = 'Home of Slartibartfast'
            else:
                (target_name, target_type, target_description) = target

            return {'Target_Identification':
                        target_identification(target_name,
                                              target_type,
                                              target_description)}
        else:
            pass
