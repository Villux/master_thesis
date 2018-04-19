
def get_label_string(kappa, theta, xi, rho):
    return f"k{kappa}_t{theta}_xi{xi}_rho{rho}"

class LabelMapper(object):
    def __init__(self):
        self.label_map = {}
        self.label_idx = 0

    def add_lable(self, kappa, theta, xi, rho):
        self.label_map[get_label_string(kappa, theta, xi, rho)] = self.label_idx
        self.label_idx += 1
        return self.label_idx - 1

    def get_label(self, kappa, theta, xi, rho):
        return self.label_map[get_label_string(kappa, theta, xi, rho)]

    def get_label_count(self):
        return len(self.label_map.keys())

if __name__ == "__main__":
    # Test
    kappas = [1,2,3]
    thetas = [4,5,6]
    xis = [7,8,9]
    rhos = [0.1, 0.2, 0.3]

    lm = LabelMapper()
    for k in kappas:
        for t in thetas:
            for x in xis:
                for r in rhos:
                    lm.add_lable(k, t, x, r)

    assert lm.get_label(1,4,7,0.1) == 0
    assert lm.get_label(1,4,7,0.1) != 1
    assert lm.get_label(3,6,9,0.3) == 80


