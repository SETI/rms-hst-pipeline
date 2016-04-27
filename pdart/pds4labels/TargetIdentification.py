from pdart.xml.Templates import *

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
            <lid_reference>urn:nasa:pds:context:target:<NODE name="lower_name"/>.<NODE name="lower_type"/></lid_reference>
            <reference_type>data_to_target</reference_type>
        </Internal_Reference>
        </Target_Identification>""")({'name': name,
              'type': type,
              'description': description,
              'lower_name': name.lower(),
              'lower_type': type.lower()});
    return func
