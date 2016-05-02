import XmlMaker


class TimeCoordinatesXmlMaker(XmlMaker.XmlMaker):
    def __init__(self, document, (start_time, end_time)):
        self.start_time = start_time
        self.end_time = end_time
        super(TimeCoordinatesXmlMaker, self).__init__(document)

    def create_xml(self, parent):
        pass
