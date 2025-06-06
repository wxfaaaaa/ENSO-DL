from .Geoformer import Geoformer

class NetFactory:
    def __init__(self):
        pass
    def net_factory(self, mypara):
        net_type = mypara.model_name
        if net_type == "Geoformer":
            net = Geoformer(mypara)
        else:
            net = None
        return net