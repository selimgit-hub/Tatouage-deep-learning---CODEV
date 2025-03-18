import hashlib
import datetime

#On organise la data dans le block

class block:
    def __init__(self,previous_block_hash,data,timestamp):
        self.previous_block_hash=previous_block_hash
        self.data=data
        self.timestamp=timestamp
        self.hash=self.get_hash()

#On crée ensuite le block d'origine
    @staticmethod
    def create_genesis_block():
        return block("0","0",datetime.datetime.now())
    
#On crée notre premiere fonction de hashage

    def get_hash(self):
        header_bin=(str(self.previous_block_hash)+str(self.data)+str(self.timestamp)).encode()
        inner_hash=hashlib.sha256(header_bin).hexdigest().encode()
        outer_hash=hashlib.sha256(inner_hash).hexdigest()
        return outer_hash
