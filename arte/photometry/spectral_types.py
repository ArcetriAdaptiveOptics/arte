



class PickelsLibrary(object):

    @classmethod
    def baseUrl(cls):
        return 'http://ssb.stsci.edu/cdbs/grid/pickles/dat_uvk/'


    @classmethod
    def filename(cls, spectralType):
        """
        Return URL to access Pickels UVKLIB spectra at STScI
        To be used with synphot.SourceSpectrum.from_file()
        """
        return "%s%s.fits" % (cls.baseUrl(),
                              cls._spectralTypeDict()[spectralType])

    @classmethod
    def _spectralTypeDict(cls):
        return {
            "O5V": "pickles_uk_1",
            "O9V": "pickles_uk_2",
            "B0V": "pickles_uk_3",
            "B1V": "pickles_uk_4",
            "B3V": "pickles_uk_5",
            "B5-7V": "pickles_uk_6",
            "B8V": "pickles_uk_7",
            "A0V": "pickles_uk_9",
            "A2V": "pickles_uk_10",
            "A3V": "pickles_uk_11",
            "A5V": "pickles_uk_12",
            "F0V": "pickles_uk_14",
            "F2V": "pickles_uk_15",
            "F5V": "pickles_uk_16",
            "F8V": "pickles_uk_20",
            "G0V": "pickles_uk_23",
            "G2V": "pickles_uk_26",
            "G5V": "pickles_uk_27",
            "G8V": "pickles_uk_30",
            "K0V": "pickles_uk_31",
            "K2V": "pickles_uk_33",
            "K3V": "pickles_uk_34",
            "K5V": "pickles_uk_36",
            "K7V": "pickles_uk_37",
            "M0V": "pickles_uk_38",
            "M2V": "pickles_uk_40",
            "M4V": "pickles_uk_43",
            "M5V": "pickles_uk_44",
            "B2IV": "pickles_uk_46",
            "B6IV": "pickles_uk_47",
            "A0IV": "pickles_uk_48",
            "A4-7IV": "pickles_uk_49",
            "F0-2IV": "pickles_uk_50",
            "F5IV": "pickles_uk_51",
            "F8IV": "pickles_uk_52",
            "G0IV": "pickles_uk_53",
            "G2IV": "pickles_uk_54",
            "G5IV": "pickles_uk_55",
            "G8IV": "pickles_uk_56",
            "K0IV": "pickles_uk_57",
            "K1IV": "pickles_uk_58",
            "K3IV": "pickles_uk_59",
            "O8III": "pickles_uk_60",
            "B1-2III": "pickles_uk_61",
            "B5III": "pickles_uk_63",
            "B9III": "pickles_uk_64",
            "A0III": "pickles_uk_65",
            "A5III": "pickles_uk_67",
            "F0III": "pickles_uk_69",
            "F5III": "pickles_uk_71",
            "G0III": "pickles_uk_72",
            "G5III": "pickles_uk_73",
            "G8III": "pickles_uk_76",
            "K0III": "pickles_uk_78",
            "K3III": "pickles_uk_87",
            "K5III": "pickles_uk_93",
            "M0III": "pickles_uk_95",
            "M5III": "pickles_uk_100",
            "M10III": "pickles_uk_105",
            "B2II": "pickles_uk_106",
            "B5II": "pickles_uk_107",
            "F0II": "pickles_uk_108",
            "F2II": "pickles_uk_109",
            "G5II": "pickles_uk_110",
            "K0-1II": "pickles_uk_111",
            "K3-4II": "pickles_uk_112",
            "M3II": "pickles_uk_113",
            "B0I": "pickles_uk_114",
            "B5I": "pickles_uk_117",
            "B8I": "pickles_uk_118",
            "A0I": "pickles_uk_119",
            "F0I": "pickles_uk_121",
            "F5I": "pickles_uk_122",
            "F8I": "pickles_uk_123",
            "G0I": "pickles_uk_124",
            "G5I": "pickles_uk_126",
            "G8I": "pickles_uk_127",
            "K2I": "pickles_uk_128",
            "K4I": "pickles_uk_130",
            "M2I": "pickles_uk_131",
        }


for k in PickelsLibrary._spectralTypeDict().keys():
    globals()[k]= PickelsLibrary.filename(k)

