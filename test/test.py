from numpy import array, cos, sin, linspace, cross, sign, arccos

s0 = array([0, 1])
Ry = lambda phi: array([[cos(phi),  sin(phi)],[-sin(phi), cos(phi)]])
s_phi = lambda phi: Ry(phi).dot(s0)
NRAY = 10

def test(phi0):
    phi = phi0 + linspace(-1e-3, 1e-3, NRAY)

    s1 = []
    for p in phi:
        s1.append(s_phi(p))
    s1 = array(s1)

    cos_psy = s1.dot(s0); sin_phi = cross(s1,s0)
    ss = sign(sin_phi)
    psy = arccos(cos_psy)*ss

    theta = psy.mean()
    print('phi0: ', phi0, 'theta: ', theta)

    return theta, s1


if __name__ == '__main__':
    from numpy import arange, pi, repeat
    from numpy.linalg import norm
    import matplotlib.pyplot as plt
    plt.ion()
    Z = [0]*NRAY
    for phi0 in [pi/8, pi/4, pi-pi/8, pi-pi/4, pi+pi/8, -pi-pi/8, -pi+pi/8, -pi/8]:
        theta, s1 = test(phi0)
        sc = sign(cos(theta))
        R = Ry(-theta)*sc
        s2 = R.dot(s1.T); s2 = s2.T
        s00 = repeat(array([s0]), NRAY, axis=0)
        plt.figure()
        plt.quiver(Z,Z,s00[:,0], s00[:,1], color='black')
        plt.quiver(Z,Z,s1[:,0], s1[:,1], color='red')
        plt.quiver(Z,Z,s2[:,0], s2[:,1], color='blue')
        plt.grid()
