# type: ignore
################################################################################
# standard_bodies module
#
# To import:
#   from standard_bodies import standard_body_identifications
#
# To use:
#   standard_body_identifications(keys)
# returns a list of Target_Identification tuples. Each tuple contains:
#   (name, alt_designations, type, description, lid)
# where:
#   name                the preferred name
#   alt_designations    a list of strings indicating alternative names.
#   type                "Planet", "Satellite", etc.
#   description         a list of strings, to be separated by newlines inside
#                       the XML Target_Identification object.
#   lid                 the LID of the object, omitting "urn:...:target:"
#
# The input argument can be a single identification or else a list of one or
# more such identifications. If multiple identifications are provided, they
# must be internally consistent, all referring to the same body. This can be
# used as an internal consistency check. For example,
#    standard_body_identifications(['Moon', 'Earth I'])
# will return the info for the Moon, but
#    standard_body_identifications(['Moon', 'Mars I'])
# will raise a ValueError, indicating that the two identifications are
# inconsistent.
#
# The reason it returns a list is for possible cases of associated objects, such
# as a planet and its moons, that might all simultaneously be targets of a given
# observation. However, this capability is not yet used for standard bodies.
#
# NOTE: Information about the complete set of standard bodies is found in
# STANDARD_BODY_INFO.py. This file must be maintained manually.
################################################################################

from . import minor_planets
from . import roman
from . import lids

MINOR_PLANET_TYPES = ("Asteroid", "Centaur", "Trans-Neptunian Object", "Dwarf Planet")
SUPPORTED_TYPES = (
    "Star",
    "Planet",
    "Satellite",
    "Ring",
    "Plasma Stream",
) + MINOR_PLANET_TYPES

################################################################################
# STANDARD_BODY_INFO
#   This is a maintained list of information about Solar System bodies. This
# list excludes small bodies unless they are in some way exceptional.
#
# Each standard body is represented by a tuple as follows:
#   (name, number, naif_id, target_type, parent, designations)
#
# Notes:
#   name            name of the body, if any.
#   number          for satellites, the satellite number; otherwise, the MPC
#                   number if it exists; otherwise, zero.
#   naif_id         NAIF body ID.
#   target_type     'Planet', 'Ring', Satellite', etc. This must be one of the
#                   standard values for the type attribute of a
#                   Target_Identification object.
#   parent          for satellites and rings, the name of the body they orbit;
#                   is empty for Sun-orbiting bodies. The parent body must also
#                   appear in this list.
#   designations    a list of zero or more technical designations for the body.
#                   These will appear as alt_designations in the
#                   Target_Identification object.
################################################################################

STANDARD_BODY_INFO = [
    ("Sun", 0, 10, "Star", "", []),
    ("Mercury", 0, 199, "Planet", "", []),
    ("Venus", 0, 299, "Planet", "", []),
    ("Earth", 0, 399, "Planet", "", []),
    ("Mars", 0, 499, "Planet", "", []),
    ("Jupiter", 0, 599, "Planet", "", []),
    ("Saturn", 0, 699, "Planet", "", []),
    ("Uranus", 0, 799, "Planet", "", []),
    ("Neptune", 0, 899, "Planet", "", []),
    ("Moon", 1, 301, "Satellite", "Earth", []),
    ("Phobos", 1, 401, "Satellite", "Mars", []),
    ("Deimos", 2, 402, "Satellite", "Mars", []),
    ("Io", 1, 501, "Satellite", "Jupiter", []),
    ("Europa", 2, 502, "Satellite", "Jupiter", []),
    ("Ganymede", 3, 503, "Satellite", "Jupiter", []),
    ("Callisto", 4, 504, "Satellite", "Jupiter", []),
    ("Amalthea", 5, 505, "Satellite", "Jupiter", []),
    ("Himalia", 6, 506, "Satellite", "Jupiter", []),
    ("Elara", 7, 507, "Satellite", "Jupiter", []),
    ("Pasiphae", 8, 508, "Satellite", "Jupiter", []),
    ("Sinope", 9, 509, "Satellite", "Jupiter", []),
    ("Lysithea", 10, 510, "Satellite", "Jupiter", []),
    ("Carme", 11, 511, "Satellite", "Jupiter", []),
    ("Ananke", 12, 512, "Satellite", "Jupiter", []),
    ("Leda", 13, 513, "Satellite", "Jupiter", []),
    ("Thebe", 14, 514, "Satellite", "Jupiter", []),
    ("Adrastea", 15, 515, "Satellite", "Jupiter", []),
    ("Metis", 16, 516, "Satellite", "Jupiter", []),
    ("Callirrhoe", 17, 517, "Satellite", "Jupiter", ["S/1999 J 1"]),
    ("Themisto", 18, 518, "Satellite", "Jupiter", ["S/1975 J 1", "S/2000 J 1"]),
    ("Magaclite", 19, 519, "Satellite", "Jupiter", []),
    ("Taygete", 20, 520, "Satellite", "Jupiter", ["S/2000 J 9"]),
    ("Chaldene", 21, 521, "Satellite", "Jupiter", ["S/2000 J 10"]),
    ("Harpalyke", 22, 522, "Satellite", "Jupiter", ["S/2000 J 5"]),
    ("Kalyke", 23, 523, "Satellite", "Jupiter", ["S/2000 J 2"]),
    ("Iocaste", 24, 524, "Satellite", "Jupiter", ["S/2000 J 3"]),
    ("Erinome", 25, 525, "Satellite", "Jupiter", ["S/2000 J 4"]),
    ("Isonoe", 26, 526, "Satellite", "Jupiter", ["S/2002 J 1"]),
    ("Praxidike", 27, 527, "Satellite", "Jupiter", ["S/2000 J 7"]),
    ("Autonoe", 28, 528, "Satellite", "Jupiter", ["S/2001 J 1"]),
    ("Thyone", 29, 529, "Satellite", "Jupiter", ["S/2001 J 2"]),
    ("Hermippe", 30, 530, "Satellite", "Jupiter", ["S/2001 J 3"]),
    ("Aitne", 31, 531, "Satellite", "Jupiter", ["S/2001 J 11"]),
    ("Eurydome", 32, 532, "Satellite", "Jupiter", ["S/2001 J 4"]),
    ("Euanthe", 33, 533, "Satellite", "Jupiter", ["S/2001 J 7"]),
    ("Euporie", 34, 534, "Satellite", "Jupiter", ["S/2001 J 10"]),
    ("Orthosie", 35, 535, "Satellite", "Jupiter", ["S/2001 J 9"]),
    ("Sponde", 36, 536, "Satellite", "Jupiter", ["S/2001 J 5"]),
    ("Kale", 37, 537, "Satellite", "Jupiter", ["S/2001 J 8"]),
    ("Pasithee", 38, 538, "Satellite", "Jupiter", ["S/2001 J 6"]),
    ("Hegemone", 39, 539, "Satellite", "Jupiter", ["S/2003 J 8"]),
    ("Mneme", 40, 540, "Satellite", "Jupiter", ["S/2003 J 21"]),
    ("Aoede", 41, 541, "Satellite", "Jupiter", ["S/2003 J 7"]),
    ("Thelxinoe", 42, 542, "Satellite", "Jupiter", ["S/2003 J 22"]),
    ("Arche", 43, 543, "Satellite", "Jupiter", ["S/2002 J 1"]),
    ("Kallichore", 44, 544, "Satellite", "Jupiter", ["S/2003 J 11"]),
    ("Helike", 45, 545, "Satellite", "Jupiter", ["S/2003 J 6"]),
    ("Carpo", 46, 546, "Satellite", "Jupiter", ["S/2003 J 20"]),
    ("Eukelade", 47, 547, "Satellite", "Jupiter", ["S/2003 J 1"]),
    ("Cyllene", 48, 548, "Satellite", "Jupiter", ["S/2003 J 13"]),
    ("Kore", 49, 549, "Satellite", "Jupiter", ["S/2003 J 14"]),
    ("", 0, 550, "Satellite", "Jupiter", ["S/2003 J 17"]),
    ("", 0, 551, "Satellite", "Jupiter", ["S/2010 J 1"]),
    ("", 0, 552, "Satellite", "Jupiter", ["S/2010 J 2"]),
    ("Dia", 53, 553, "Satellite", "Jupiter", ["S/2000 J 11"]),
    ("", 0, 554, "Satellite", "Jupiter", ["S/2016 J 1"]),
    ("", 0, 555, "Satellite", "Jupiter", ["S/2003 J 18"]),
    ("", 0, 556, "Satellite", "Jupiter", ["S/2011 J 2"]),
    ("", 0, 557, "Satellite", "Jupiter", ["S/2003 J 5"]),
    ("", 0, 558, "Satellite", "Jupiter", ["S/2003 J 15"]),
    ("", 0, 55060, "Satellite", "Jupiter", ["S/2003 J 2"]),
    ("", 0, 55061, "Satellite", "Jupiter", ["S/2003 J 3"]),
    ("", 0, 55062, "Satellite", "Jupiter", ["S/2003 J 4"]),
    ("", 0, 55064, "Satellite", "Jupiter", ["S/2003 J 9"]),
    ("", 0, 55065, "Satellite", "Jupiter", ["S/2003 J 10"]),
    ("", 0, 55066, "Satellite", "Jupiter", ["S/2003 J 12"]),
    ("", 0, 55068, "Satellite", "Jupiter", ["S/2003 J 16"]),
    ("", 0, 55070, "Satellite", "Jupiter", ["S/2003 J 19"]),
    ("", 0, 55071, "Satellite", "Jupiter", ["S/2003 J 23"]),
    ("", 0, 55074, "Satellite", "Jupiter", ["S/2011 J 1"]),
    ("Mimas", 1, 601, "Satellite", "Saturn", []),
    ("Enceladus", 2, 602, "Satellite", "Saturn", []),
    ("Tethys", 3, 603, "Satellite", "Saturn", []),
    ("Dione", 4, 604, "Satellite", "Saturn", []),
    ("Rhea", 5, 605, "Satellite", "Saturn", []),
    ("Titan", 6, 606, "Satellite", "Saturn", []),
    ("Hyperion", 7, 607, "Satellite", "Saturn", []),
    ("Iapetus", 8, 608, "Satellite", "Saturn", []),
    ("Phoebe", 9, 609, "Satellite", "Saturn", []),
    ("Janus", 10, 610, "Satellite", "Saturn", []),
    ("Epimetheus", 11, 611, "Satellite", "Saturn", []),
    ("Helene", 12, 612, "Satellite", "Saturn", ["S/1980 S 6"]),
    ("Telesto", 13, 613, "Satellite", "Saturn", ["S/1980 S 13"]),
    ("Calypso", 14, 614, "Satellite", "Saturn", ["S/1980 S 25"]),
    ("Atlas", 15, 615, "Satellite", "Saturn", ["S/1980 S 28"]),
    ("Prometheus", 16, 616, "Satellite", "Saturn", ["S/1980 S 27"]),
    ("Pandora", 17, 617, "Satellite", "Saturn", ["S/1980 S 26"]),
    ("Pan", 18, 618, "Satellite", "Saturn", ["S/1981 S 13"]),
    ("Ymir", 19, 619, "Satellite", "Saturn", ["S/2000 S 1"]),
    ("Paaliaq", 20, 620, "Satellite", "Saturn", ["S/2000 S 2"]),
    ("Tarvos", 21, 621, "Satellite", "Saturn", ["S/2000 S 4"]),
    ("Ijiraq", 22, 622, "Satellite", "Saturn", ["S/2000 S 6"]),
    ("Suttungr", 23, 623, "Satellite", "Saturn", ["S/2000 S 12"]),
    ("Kiviuq", 24, 624, "Satellite", "Saturn", ["S/2000 S 5"]),
    ("Mundilfari", 25, 625, "Satellite", "Saturn", ["S/2000 S 9"]),
    ("Albiorix", 26, 626, "Satellite", "Saturn", ["S/2000 S 11"]),
    ("Skathi", 27, 627, "Satellite", "Saturn", ["S/2000 S 8"]),
    ("Erriapus", 28, 628, "Satellite", "Saturn", ["S/2000 S 10"]),
    ("Siarnaq", 29, 629, "Satellite", "Saturn", ["S/2000 S 3"]),
    ("Thrymr", 30, 630, "Satellite", "Saturn", ["S/2000 S 7"]),
    ("Narvi", 31, 631, "Satellite", "Saturn", ["S/2003 S 1"]),
    ("Methone", 32, 632, "Satellite", "Saturn", ["S/2004 S 1"]),
    ("Pallene", 33, 633, "Satellite", "Saturn", ["S/2004 S 2"]),
    ("Polydeuces", 34, 634, "Satellite", "Saturn", ["S/2004 S 5"]),
    ("Daphnis", 35, 635, "Satellite", "Saturn", ["S/2005 S 1"]),
    ("Aegir", 36, 636, "Satellite", "Saturn", ["S/2004 S 10"]),
    ("Bebhionn", 37, 637, "Satellite", "Saturn", ["S/2004 S 11"]),
    ("Bergelmir", 38, 638, "Satellite", "Saturn", ["S/2004 S 15"]),
    ("Bestla", 39, 639, "Satellite", "Saturn", ["S/2004 S 18"]),
    ("Farbauti", 40, 640, "Satellite", "Saturn", ["S/2004 S 9"]),
    ("Fenrir", 41, 641, "Satellite", "Saturn", ["S/2004 S 16"]),
    ("Fornjot", 42, 642, "Satellite", "Saturn", ["S/2004 S 8"]),
    ("Hati", 43, 643, "Satellite", "Saturn", ["S/2004 S 14"]),
    ("Hyrrokkin", 44, 644, "Satellite", "Saturn", []),
    ("Kari", 45, 645, "Satellite", "Saturn", ["S/2006 S 2"]),
    ("Loge", 46, 646, "Satellite", "Saturn", ["S/2006 S 5"]),
    ("Skoll", 47, 647, "Satellite", "Saturn", ["S/2006 S 8"]),
    ("Surtur", 48, 648, "Satellite", "Saturn", ["S/2006 S 7"]),
    ("Anthe", 49, 649, "Satellite", "Saturn", ["S/2007 S 4"]),
    ("Jarnsaxa", 50, 650, "Satellite", "Saturn", ["S/2006 S 6"]),
    ("Greip", 51, 651, "Satellite", "Saturn", ["S/2006 S 4"]),
    ("Tarqeq", 52, 652, "Satellite", "Saturn", ["S/2007 S 1"]),
    ("Aegaeon", 53, 653, "Satellite", "Saturn", ["S/2008 S 1"]),
    ("", 0, 65035, "Satellite", "Saturn", ["S/2004 S 7"]),
    ("", 0, 65040, "Satellite", "Saturn", ["S/2004 S 12"]),
    ("", 0, 65041, "Satellite", "Saturn", ["S/2004 S 13"]),
    ("", 0, 65045, "Satellite", "Saturn", ["S/2004 S 17"]),
    ("", 0, 65048, "Satellite", "Saturn", ["S/2006 S 1"]),
    ("", 0, 65050, "Satellite", "Saturn", ["S/2006 S 3"]),
    ("", 0, 65055, "Satellite", "Saturn", ["S/2007 S 2"]),
    ("", 0, 65056, "Satellite", "Saturn", ["S/2007 S 3"]),
    ("Ariel", 1, 701, "Satellite", "Uranus", []),
    ("Umbriel", 2, 702, "Satellite", "Uranus", []),
    ("Titania", 3, 703, "Satellite", "Uranus", []),
    ("Oberon", 4, 704, "Satellite", "Uranus", []),
    ("Miranda", 5, 705, "Satellite", "Uranus", []),
    ("Cordelia", 6, 706, "Satellite", "Uranus", ["S/1986 U 7"]),
    ("Ophelia", 7, 707, "Satellite", "Uranus", ["S/1986 U 8"]),
    ("Bianca", 8, 708, "Satellite", "Uranus", ["S/1986 U 9"]),
    ("Cressida", 9, 709, "Satellite", "Uranus", ["S/1986 U 3"]),
    ("Desdemona", 10, 710, "Satellite", "Uranus", ["S/1986 U 6"]),
    ("Juliet", 11, 711, "Satellite", "Uranus", ["S/1986 U 2"]),
    ("Portia", 12, 712, "Satellite", "Uranus", ["S/1986 U 1"]),
    ("Rosalind", 13, 713, "Satellite", "Uranus", ["S/1986 U 4"]),
    ("Belinda", 14, 714, "Satellite", "Uranus", ["S/1986 U 5"]),
    ("Puck", 15, 715, "Satellite", "Uranus", ["S/1985 U 1"]),
    ("Caliban", 16, 716, "Satellite", "Uranus", ["S/1997 U 1"]),
    ("Sycorax", 17, 717, "Satellite", "Uranus", ["S/1997 U 2"]),
    ("Prospero", 18, 718, "Satellite", "Uranus", ["S/1999 U 3"]),
    ("Setebos", 19, 719, "Satellite", "Uranus", ["S/1999 U 1"]),
    ("Stephano", 20, 720, "Satellite", "Uranus", ["S/1999 U 2"]),
    ("Trinculo", 21, 721, "Satellite", "Uranus", ["S/2001 U 1"]),
    ("Francisco", 22, 722, "Satellite", "Uranus", ["S/2001 U 3"]),
    ("Margaret", 23, 723, "Satellite", "Uranus", ["S/2003 U 3"]),
    ("Ferdinand", 24, 724, "Satellite", "Uranus", []),
    ("Perdita", 25, 725, "Satellite", "Uranus", ["S/1986 U 10"]),
    ("Mab", 26, 726, "Satellite", "Uranus", ["S/2003 U 1"]),
    ("Cupid", 27, 727, "Satellite", "Uranus", ["S/2003 U 2"]),
    ("Triton", 1, 801, "Satellite", "Neptune", []),
    ("Nereid", 2, 802, "Satellite", "Neptune", []),
    ("Naiad", 3, 803, "Satellite", "Neptune", []),
    ("Thalassa", 4, 804, "Satellite", "Neptune", []),
    ("Despina", 5, 805, "Satellite", "Neptune", []),
    ("Galatea", 6, 806, "Satellite", "Neptune", []),
    ("Larissa", 7, 807, "Satellite", "Neptune", []),
    ("Proteus", 8, 808, "Satellite", "Neptune", []),
    ("Halimede", 9, 809, "Satellite", "Neptune", ["S/2002 N 1"]),
    ("Psamathe", 10, 810, "Satellite", "Neptune", ["S/2003 N 1"]),
    ("Sao", 11, 811, "Satellite", "Neptune", ["S/2002 N 2"]),
    ("Laomedeia", 12, 812, "Satellite", "Neptune", ["S/2002 N 3"]),
    ("Neso", 13, 813, "Satellite", "Neptune", ["S/2003 N 2"]),
    ("Hippocamp", 14, 814, "Satellite", "Neptune", ["S/2004 N 1"]),
    # Dwarf planet systems
    # Bbelow, "$" in a designation is a placeholder for the parent designation,
    # e.g., "S/2005 $ 1" -> "S/2005 (136199) 1" or "S/2005 (2003 UB313) 1"
    ("Ceres", 1, 2000001, "Dwarf Planet", "", ["1899 OF", "1943 XB"]),
    ("Pluto", 134340, 999, "Dwarf Planet", "", []),
    ("Charon", 1, 901, "Satellite", "Pluto", ["S/1978 P 1", "P1"]),
    ("Nix", 2, 902, "Satellite", "Pluto", ["S/2005 $ 2", "S/2005 P 2", "P2"]),
    ("Hydra", 3, 903, "Satellite", "Pluto", ["S/2005 $ 1", "S/2005 P 1", "P3"]),
    ("Kerberos", 4, 904, "Satellite", "Pluto", ["S/2011 $ 1", "S/2011 P 1", "P4"]),
    ("Styx", 5, 905, "Satellite", "Pluto", ["S/2012 $ 1", "S/2012 P 1", "P5"]),
    ("Eris", 136199, 2136199, "Dwarf Planet", "", ["2003 UB313"]),
    ("Dysnomia", 1, 0, "Satellite", "Eris", ["S/2005 $ 1"]),
    ("Haumea", 136108, 2136108, "Dwarf Planet", "", ["2003 EL61"]),
    ("Hi'iaka", 1, 0, "Satellite", "Haumea", ["S/2005 $ 1"]),
    ("Namaka", 2, 0, "Satellite", "Haumea", ["S/2005 $ 2"]),
    ("Makemake", 136472, 2136472, "Dwarf Planet", "", ["2005 FY9"]),
    ("", 1, 0, "Satellite", "Makemake", ["S/2015 $ 1"]),
    # Other minor planet satellites
    ("Ida", 243, 2000243, "Asteroid", "", ["1910 CD", "1988 DB1"]),
    ("Dactyl", 1, 2431011, "Satellite", "Ida", ["S/1993 $ 1"]),
    ("Quaoar", 50000, 2050000, "Trans-Neptunian Object", "", ["2002 LM60"]),
    ("Weywot", 1, 0, "Satellite", "Quaoar", ["S/2006 $ 1"]),
    ("Gonggong", 225088, 2225088, "Trans-Neptunian Object", "", ["2007 OR10"]),
    ("Xiangliu", 1, 0, "Satellite", "Gonggong", ["S/2010 $ 1"]),
    ("Salacia", 120347, 2120347, "Trans-Neptunian Object", "", ["2004 SB60"]),
    ("Actaea", 1, 0, "Satellite", "Salacia", ["S/2006 $ 1"]),
    ("Orcus", 90482, 2090482, "Trans-Neptunian Object", "", ["2004 DW"]),
    ("Vanth", 1, 0, "Satellite", "Orcus", ["S/2005 $ 1"]),
    # Rings (real and imagined)
    ("Mars Rings", 0, 0, "Ring", "Mars", []),
    ("Jupiter Rings", 0, 0, "Ring", "Jupiter", []),
    ("Saturn Rings", 0, 0, "Ring", "Saturn", []),
    ("Uranus Rings", 0, 0, "Ring", "Uranus", []),
    ("Neptune Rings", 0, 0, "Ring", "Neptune", []),
    ("Pluto Rings", 0, 0, "Ring", "Pluto", []),
    # Io Torus
    ("Io Torus", 0, 0, "Plasma Stream", "Jupiter", []),
]

################################################################################
# Class for information about standard bodies
################################################################################


class StandardBodyInfo(object):

    ######## CONSTRUCTOR

    def __init__(self, name, number, naif_id, body_type, parent, designations=[]):

        self.name = name
        self.number = number
        self.naif_id = naif_id
        self.body_type = body_type
        self.parent = parent
        self.designations = designations

        self.parent_info_filled = None  # Property for satellites, rings
        self.minor_planet_info_filled = None  # Property for minor planets

        assert self.body_type in SUPPORTED_TYPES, (
            "Unsupported target type: " + self.body_type
        )

        if self.body_type == "Star" and self.name != "Sun":
            raise ValueError("Stars other than the Sun are not supported")

    @property
    def parent_info(self):

        if not self.parent_info_filled:
            self.parent_info_filled = STANDARD_BODY_LOOKUP[self.parent]

        return self.parent_info_filled

    @property
    def minor_planet_info(self):

        if not self.minor_planet_info_filled and self.body_type in MINOR_PLANET_TYPES:
            if self.designations:
                mp = minor_planets.MinorPlanetInfo(
                    self.body_type,
                    self.number,
                    self.designations[0],
                    self.name,
                    self.designations[1:],
                )
            else:
                mp = minor_planets.MinorPlanetInfo(
                    self.body_type, self.number, "", self.name, []
                )

            mp.naif_id = self.naif_id
            self.minor_planet_info_filled = mp

        return self.minor_planet_info_filled

    ######## UTILITIES

    def __str__(self):
        return (
            "StandardBodyInfo("
            + self.name
            + "|"
            + str(self.number)
            + "|"
            + str(self.naif_id)
            + "|"
            + self.body_type
            + "|"
            + self.parent
            + "|"
            + str(self.designations)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):

        # Matching names confirm equality
        if self.name and other.name:
            return self.name.lower() == other.name.lower()

        # Matching NAIF IDs confirm equality
        if self.naif_id and other.naif_id:
            return self.naif_id == other.naif_id

        # Matching designations confirm equality
        if self.designations and other.designations:
            return (
                self.parent == other.parent and self.designations == other.designations
            )

        return False

    def merge(self, other):

        self.name = self.name or other.name
        self.number = self.number or other.number
        self.naif_id = self.naif_id or other.naif_id
        self.body_type = self.body_type or other.body_type
        self.parent = self.parent or other.parent
        self.parent_info = self.parent_info or other.parent_info

        # Merge lists of designations
        for designation in other.designations:
            if designation not in self.designations:
                self.designations.append(designation)

    def copy(self, deep=False):

        clone = StandardBodyInfo(
            self.name,
            self.number,
            self.naif_id,
            self.body_type,
            self.parent,
            self.designations,
        )

        if deep:
            clone.designations = list(clone.designations)

        return clone

    def full_names(self):

        if self.body_type in ("Star", "Planet", "Ring", "Plasma Stream"):
            names = [self.name]

        elif self.body_type == "Satellite":
            parent = self.parent_info

            if self.number:
                xx = roman.int_to_roman(self.number)

            if parent.body_type == "Planet":
                names = []
                if self.name:
                    names += [
                        self.name,
                        f"{parent.name} {xx} ({self.name})",
                        f"{parent.name} {xx} {self.name}",
                    ]
                    names += [f"{n} ({self.name})" for n in self.designations]

                names += self.designations

                if self.number:
                    names += [f"{parent.name} {xx}", f"{parent.name[0]}{self.number}"]

            else:
                names = []
                if self.name:
                    names += [
                        f"{parent.number} {parent.name} {xx} {self.name}",
                        f"({parent.number}) {parent.name} {xx} {self.name}",
                        f"{parent.number} {parent.name} {xx} ({self.name})",
                        f"{parent.name} {xx} ({self.name})",
                        f"{parent.name} {xx} {self.name}",
                        self.name,
                    ]

                if self.number:
                    names += [
                        f"{parent.number} {parent.name} {xx}",
                        f"({parent.number}) {parent.name} {xx}",
                        f"{parent.name} {xx}",
                    ]

                for designation in self.designations:
                    if "$" in designation:
                        for pname in [
                            f"({parent.number})",
                            f"({parent.number}) {parent.name}",
                            f"{parent.name}",
                        ]:
                            names += [designation.replace("$", pname)]
                    else:
                        names += [designation]

        elif self.minor_planet_info:
            return self.minor_planet_info.full_names()

        else:
            raise ValueError(f"Unsupported type: {self.body_type}")

        if self.naif_id:
            names += [f"NAIF ID {self.naif_id}"]

        return names

    def lid(self):

        if self.body_type in ("Star", "Planet", "Plasma Stream"):
            string = self.name

        elif self.body_type == "Ring":
            string = self.parent + ".rings"

        elif self.body_type == "Satellite":
            parent = self.parent_info

            if parent.body_type == "Planet":
                parent_name = parent.name
            else:
                parent_name = str(parent.number) + "_" + parent.name

            if self.name:
                body_name = self.name
            elif parent.body_type == "Planet":
                body_name = self.designations[0].replace("/", "").replace(" ", "")
            elif "$" in self.designations[0]:
                # Change, e.g., "S/2005 (136108) 1" to "S2005_1"
                body_name = self.designations[0].replace("$ ", "")
            else:
                body_name = self.designations[0]

            string = parent_name + "." + body_name

        elif self.minor_planet_info:
            return self.minor_planet_info.lid()

        return lids.clean(self.body_type + "." + string)

    def target_identifications(self):

        if self.body_type in ("Star", "Planet", "Plasma Stream"):
            return [(self.name, [], self.body_type, [], self.lid())]

        if self.body_type == "Ring":
            return [
                (
                    self.name,
                    [],
                    "Ring",
                    ["Any or all of the rings of " + self.parent],
                    self.lid(),
                )
            ]

        if self.body_type == "Satellite":
            names = self.full_names()
            return [
                (
                    names[0],
                    names[1:],
                    "Satellite",
                    [
                        f"Satellite of: {self.parent_info.full_names()[0]}",
                        f"Type of primary: {self.parent_info.body_type}",
                        f"LID of primary: {self.parent_info.lid()}",
                        f"NAIF ID of primary: {self.parent_info.naif_id}",
                    ],
                    self.lid(),
                )
            ]

        if self.minor_planet_info:
            return self.minor_planet_info.target_identifications()

        raise ValueError(f"Unsupported type: {self.body_type}")


################################################################################
# Create a dictionary from the STANDARD_BODY_INFO list. Key it by every
# conceivable string
################################################################################

STANDARD_BODIES = []
STANDARD_BODY_LOOKUP = {}

for (name, number, naif_id, target_type, parent, designations) in STANDARD_BODY_INFO:

    info = StandardBodyInfo(name, number, naif_id, target_type, parent, designations)
    STANDARD_BODIES.append(info)

    lid = info.lid()
    suffix = lid.partition("target:")[2]

    names = info.full_names() + [info.name, lid, suffix]
    for name in names:
        if name:
            STANDARD_BODY_LOOKUP[name] = info
            STANDARD_BODY_LOOKUP[name.upper()] = info

        # If a minor planet...
        if info.number and info.body_type != "Satellite":
            STANDARD_BODY_LOOKUP[str(info.number)] = info


def standard_body_identifications(keys, include=[]):

    if isinstance(keys, (str, int)):
        keys = [keys]

    if standard_body_identifications.DEBUG:
        print("standard body keys:", keys)

    body = None
    for key in keys:
        key = str(key).upper()

        # Name not found
        if key not in STANDARD_BODY_LOOKUP:
            if standard_body_identifications.IGNORE_EXTRA_NAMES:
                print(f"WARNING: Ignored standard body identifier: {key}")
                if standard_body_identifications.DEBUG:
                    print("unknown standard body identifier, ignored:", key)
            else:
                raise ValueError(f"Unrecognized standard body: {key}")

        test_body = STANDARD_BODY_LOOKUP[key]

        # Save the first match
        if not body:
            body = test_body
            if standard_body_identifications.DEBUG:
                print("standard body identified:", key)

        # Subsequent names must match the first
        if body != test_body:
            raise ValueError(f"Inconsistent standard body identifiers: {keys}")

    # Raises KeyError on failure
    if not body:
        raise KeyError(f"No matching standard bodies: {keys}")

    # Create the return list
    info = [body]

    include = set([word.lower() for word in include])

    if "parent" in include:
        include.remove("parent")
        if body.parent_info:
            info += [body.parent_info]

    # This could be a long list for planets. More sensible for dwarf planets.
    if "satellites" in include:
        include.remove("satellites")
        info += [
            s
            for s in STANDARD_BODIES
            if body.name == s.parent and s.body_type == "Satellite"
        ]

    if "rings" in include:
        include.remove("rings")
        info += [
            r
            for r in STANDARD_BODIES
            if body.name == r.parent and r.body_type == "Ring"
        ]

    if "tori" in include:
        include.remove("tori")
        info += [
            t
            for t in STANDARD_BODIES
            if body.name == t.parent and t.body_type == "Plasma Stream"
        ]

    if include:
        raise ValueError("Unrecognized include type: " + str(include))

    results = []
    for body in info:
        results += body.target_identifications()

    return results


# These attributes of the function control its behavior...

# True to ignore unrecognized but superfluous names
standard_body_identifications.IGNORE_EXTRA_NAMES = False

# True to print debugging info
standard_body_identifications.DEBUG = False

################################################################################
