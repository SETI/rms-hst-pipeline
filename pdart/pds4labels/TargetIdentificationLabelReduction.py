from pdart.exceptions.Combinators import *
from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


# For product labels: produces the Target_Identification element.

def target_identification(target_name, target_type, target_description):
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
            'name': target_name,
            'type': target_type,
            'description': target_description,
            'lower_name': target_name.lower(),
            'lower_type': target_type.lower()})
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
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


def _get_placeholder_target(*args, **kwargs):
    return ('Magrathea', 'Planet', 'Home of Slartibartfast')


_get_target = multiple_implementations('_get_target',
                                       _get_target_from_header_unit,
                                       _get_placeholder_target)


def _db_get_target_from_header_unit(conn, lid):
    with closing(conn.cursor()) as cursor:
        cursor.execute(
            """SELECT value FROM cards
               WHERE product=? AND hdu_index=0 AND keyword='TARGNAME'""")
        (targname,) = cursor.fetchone()

    for prefix, (name, type) in _approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_db_target = multiple_implementations('_get_db_target',
                                          _db_get_target_from_header_unit,
                                          _get_placeholder_target)


def get_db_target(conn, lid):
    return target_identification(*(_get_db_target(conn, lid)))


class TargetIdentificationLabelReduction(Reduction):
    """Reduce a product to an XML Target_Identification node template."""
    def reduce_fits_file(self, file, get_reduced_hdus):
        # Doc -> Node
        get_target = multiple_implementations('get_target',
                                              lambda: get_reduced_hdus()[0],
                                              _get_placeholder_target)
        return target_identification(*get_target())

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # tuple or None
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        # tuple or None
        if n == 0:
            return _get_target(header_unit)
        else:
            pass
