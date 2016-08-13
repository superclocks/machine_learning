
class Str2IdMap():
    def __init__(self):
        self._to_id = {}
        self._to_str = []
    def getStr(self, id):
        return self._to_str[id]
    def getId(self, str):
        if(self._to_id.has_key(str) == False):
            id = len(self._to_id.items())
            self._to_id[str] = id
            self._to_str.append(str)
            return id
        else:
            return self._to_id[str]