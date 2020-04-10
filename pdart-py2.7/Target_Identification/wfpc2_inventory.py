import oops
import oops.inst.hst.wfpc2 as wfpc2

PLANETS = ('MARS', 'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO')

oops.define_small_body(1000099, '8P-TUTTLE', spk='/Users/mark/Desktop/Asteroid-SPKs/Binary/8PTuttle_1_1858_A1.bsp')

families = {}
included = set(['EARTH', 'SUN'])    # always ignore Earth and Sun

for planet in PLANETS:
    body = oops.Body.lookup(planet)
    children = body.select_children(include_all=['SATELLITE'])
    families[planet] = [planet] + [c.name for c in children]
    included |= set(families[planet])

extras = [b for b in oops.Body.BODY_REGISTRY.values()]
extras = oops.Body.keywords_do_not_include(extras, ['BARYCENTER', 'RING'])
names = [x.name for x in extras if x.name not in included]

filepath = '/Volumes/Data-SSD/PDART/8P-Tuttle/ua320102m_d0m.fits'
filepath = '/Volumes/Data-SSD/Astronomy/Mars/HST-08579/Visit02/u644020cr_d0m.fits'

inventory = []
for ccd in (1,2,3,4):
    try:
        obs = wfpc2.from_file(filepath, ccd=ccd, astrometry=True)
    except ValueError:
        continue

    for key in families:
        try:
            inventory += obs.inventory(families[key])
        except RuntimeError:
            pass

    for name in names:
        try:
            inventory += obs.inventory([name])
        except RuntimeError:
            pass

print inventory
