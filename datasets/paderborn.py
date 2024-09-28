from datasets.base_dataset import BaseDataset

class Paderborn(BaseDataset):    
    
    def list_of_bearings(self):
        return [
        ("K001.rar", "K001.rar"), ("K002.rar", "K002.rar"), ("K003.rar", "K003.rar"), ("K004.rar", "K004.rar"), ("K005.rar", "K005.rar"), ("K006.rar", "K006.rar"), 
        ("KA01.rar", "KA01.rar"), ("KA03.rar", "KA03.rar"), ("KA04.rar", "KA04.rar"), ("KA05.rar", "KA05.rar"), ("KA06.rar", "KA06.rar"), ("KA07.rar", "KA07.rar"), ("KA09.rar", "KA09.rar"), ("KA15.rar", "KA15.rar"), ("KA16.rar", "KA16.rar"), ("KA22.rar", "KA22.rar"), ("KA30.rar", "KA30.rar"), 
        ("KI01.rar", "KI01.rar"), ("KI03.rar", "KI03.rar"), ("KI04.rar", "KI04.rar"), ("KI05.rar", "KI05.rar"), ("KI07.rar", "KI07.rar"), ("KI08.rar", "KI08.rar"), ("KI14.rar", "KI14.rar"), ("KI16.rar", "KI16.rar"), ("KI17.rar", "KI17.rar"), ("KI18.rar", "KI18.rar"), ("KI21.rar", "KI21.rar"), 
        ]

    def __init__(self):
        super().__init__(rawfilesdir = "data/raw/paderborn", 
                         url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/")
    
    def __str__(self):
        return "Paderborn"