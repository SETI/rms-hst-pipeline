from pdart.exceptions.Combinators import *
from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


# For product labels: produces the Target_Identification element.

def target_identification(name, type, description):
    """
    Given a target name and target type, return a function that takes
    a document and returns a filled-out Target_Identification XML
    node, used in product labels.
    """
    func = interpret_template("""<Target_Identification>
        <name><NODE name="name"/></name>
        <type><NODE name="type"/></type>
        <description><NODE name="description"/></description>
        <Internal_Reference>
            <lid_reference>urn:nasa:pds:context:target:\
<NODE name="lower_name"/>.<NODE name="lower_type"/></lid_reference>
            <reference_type>data_to_target</reference_type>
        </Internal_Reference>
        </Target_Identification>""")({
            'name': name,
            'type': type,
            'description': description,
            'lower_name': name.lower(),
            'lower_type': type.lower()})
    return func


_approximate_target_table = {
    'JUP': ('Jupiter', 'Planet'),
    'SAT': ('Saturn', 'Planet'),
    'URA': ('Uranus', 'Planet'),
    'NEP': ('Neptune', 'Planet'),
    'PLU': ('Pluto', 'Dwarf Planet'),
    'PLCH': ('Pluto', 'Dwarf Planet'),
    'IO': ('Io', 'Satellite'),
    'EUR': ('Europa', 'Satellite'),
    'GAN': ('Ganymede', 'Satellite'),
    'CALL': ('Callisto', 'Satellite'),
    'TITAN': ('Titan', 'Satellite'),
    'TRIT': ('Triton', 'Satellite'),
    'DIONE': ('Dione', 'Satellite'),
    'IAPETUS': ('Iapetus', 'Satellite')
    }


def _get_target_from_header_unit(header_unit):
    targname = header_unit['TARGNAME']
    for prefix, (name, type) in _approximate_target_table.iteritems():
        if targname.startswith(prefix):
            desc = 'The %s %s' % (type.lower(), name)
            return (name, type, desc)
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


def _get_placeholder_target(*args, **kwargs):
    target_name = 'Magrathea'
    target_type = 'Planet'
    target_description = 'Home of Slartibartfast'
    return (target_name, target_type, target_description)


_get_target = multiple_implementations('get_target',
                                       _get_target_from_header_unit,
                                       _get_placeholder_target)


class TargetIdentificationLabelReduction(Reduction):
    """Reduce a product to an XML Target_Identification node template."""
    def reduce_fits_file(self, file, get_reduced_hdus):
        # Doc -> Node
        reduced_hdus = get_reduced_hdus()
        return reduced_hdus[0]

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # Doc -> Node or None
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        # Doc -> Node or None
        if n == 0:
            (target_name,
             target_type,
             target_description) = _get_target(header_unit)
            return target_identification(target_name,
                                         target_type,
                                         target_description)
        else:
            pass
