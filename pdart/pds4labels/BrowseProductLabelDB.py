from contextlib import closing

from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.BrowseProductLabelXml import *


def make_db_browse_product_labels(conn, archive):
    """
    Given a database connection and an :class:`pdart.pds4.Archive`,
    create PDS4 labels for the browse products.
    """

    # TODO Polish (with SQL joins?).  First, an inefficient
    # implementation.
    with closing(conn.cursor()) as cursor:
        colls = list(cursor.execute(
                """SELECT collection, bundle, suffix FROM collections
                   WHERE prefix='data' AND suffix='raw'"""))

        for (c, b, suffix) in colls:
            cursor.execute('SELECT proposal_id FROM bundles WHERE bundle=?',
                           (b,))
            (proposal_id,) = cursor.fetchone()
            print '>>>>', c
            prods = [p for (p,)
                     in cursor.execute(
                    """SELECT product FROM products WHERE collection=?
                       EXCEPT SELECT product FROM bad_fits_files""",
                    (c,))]
            for p in prods:
                lid = LID(p)
                product = Product(archive, lid)
                browse_product = product.browse_product()
                browse_file_name = lid.product_id + '.jpg'
                browse_image_file = list(browse_product.files())[0]
                object_length = os.path.getsize(
                    browse_image_file.full_filepath())
                label = make_label({
                        'proposal_id': str(proposal_id),
                        'suffix': suffix,
                        'browse_lid': str(browse_product.lid),
                        'data_lid': str(lid),
                        'browse_file_name': browse_file_name,
                        'object_length': str(object_length)
                        }).toxml()

                label_fp = browse_product.label_filepath()

                with open(label_fp, 'w') as f:
                    f.write(label)
