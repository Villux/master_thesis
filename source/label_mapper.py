
def get_label_string(kappa, theta, xi, rho):
    return f"k{kappa}_t{theta}_xi{xi}_rho{rho}"

class LabelMapper(object):
    def __init__(self, kappa, theta, xi, rho):
        self.label_map = {}
        label_idx = 0
        for k in kappa:
            for t in theta:
                for x in xi:
                    for r in rho:
                        self.label_map[get_label_string(k, t, x, r)] = label_idx
                        label_idx += 1

    def get_label(self, kappa, theta, xi, rho):
        return self.label_map[get_label_string(kappa, theta, xi, rho)]

if __name__ == "__main__":
    # Test
    kappas = [1,2,3]
    thetas = [4,5,6]
    xis = [7,8,9]
    rhos = [0.1, 0.2, 0.3]

    lm = LabelMapper(kappas, thetas, xis, rhos)

    assert lm.get_label(1,4,7,0.1) == 0
    assert lm.get_label(1,4,7,0.1) != 1
    assert lm.get_label(3,6,9,0.3) == 80


