# -*- coding: utf-8 -*-
'''
Ala multi-segmento trapezoidale - Appendice B.1 del manuale Genair.
Autonomo: adatta da solo i profili alari.

Uso (dalla shell "Welcome to Genair!", che si trova in play/):
    >>> %run trapezoidal_wing.py
'''

import numpy as np
import nurbs

from nurbs.point   import Point
from geom.airfoil  import Airfoil
from geom.io       import save, save_IGES
from geom.wing     import Wing, WingMerger, BSplineFunctionCreator1
from plot.figure   import Figure as draw

# ------------------------------------------------------------------
# 1) PARAMETRI DELLA PIANFORMA
#    N stazioni: AIRFOIL/TWIST/CHORD hanno N elementi,
#    SPAN/DIHEDRAL/SWEEP ne hanno N-1 (uno per segmento)
# ------------------------------------------------------------------
AIRFOIL  = ['generated_LDS-2_AIRFOIL_lds2-il.dat', 'generated_LDS-2_AIRFOIL_lds2-il.dat', 'generated_LDS-2_AIRFOIL_lds2-il.dat']
TWIST    = [0, 10, 40]              # torsione [gradi] per stazione
CHORD    = [5.82, 3.40, 0.78]     # corda per stazione [m]
SPAN     = [4.66, 8.44]           # apertura di ogni segmento [m]
DIHEDRAL = [2, 5]                 # diedro di ogni segmento [gradi]
SWEEP    = [30, 20]               # freccia di ogni segmento [gradi]
REF      = 'LE'                   # curva per lo sweep: 'LE', 'QC' o 'TE'
K_FIT    = 1                      # alza a 5-10 se usi twist non nullo

# ------------------------------------------------------------------
# 2) CARICAMENTO E ADATTAMENTO DEI PROFILI (corda 1, QC nell'origine)
#    Nessuna finestra intermedia; sharpen solo se davvero necessario
# ------------------------------------------------------------------
def build_airfoil(fn):
    a = Airfoil('airfoils/' + fn)
    a.fit(show=False)
    if not a.issharp:
        try:
            a.sharpen(show=False)
        except Exception:
            a.sharpen2(show=False)
        a.fit(show=False)
    a.transform(show=False)
    return a

airs = [build_airfoil(fn) for fn in AIRFOIL]

# ------------------------------------------------------------------
# 3) CLASSE TRAPEZOIDAL WING (dal manuale, getattr al posto di eval)
# ------------------------------------------------------------------
class TrapezoidalWing(Wing):

    def __init__(self, a0, a1, t0, t1, c0, c1, s, d, p):
        super(TrapezoidalWing, self).__init__(a0, a1)
        tw = BSplineFunctionCreator1(end=(t0, t1)).fit()
        sc = BSplineFunctionCreator1(end=(c0, c1)).fit()
        T  = nurbs.tb.make_linear_curve(Point(), Point(y=s))
        self.orient(T, Tw=tw, show=False)
        self.fit(K=K_FIT, scs=(sc, None, sc), show=False)
        if REF != 'QC':
            self.T = getattr(self, REF)
        self.glue(); self.dihedral = d; self.sweep = p

# ------------------------------------------------------------------
# 4) GENERAZIONE E FUSIONE DEI SEGMENTI
#    (merge(show=False): il WingMerger trasla da solo i segmenti)
# ------------------------------------------------------------------
ws = []
for i in range(len(airs) - 1):
    w = TrapezoidalWing(airs[i], airs[i+1],
                        TWIST[i],  TWIST[i+1],
                        CHORD[i],  CHORD[i+1],
                        SPAN[i],   DIHEDRAL[i], SWEEP[i])
    ws.append(w)

multi_segmented_wing = WingMerger(*ws)
multi_segmented_wing.merge(show=False)

# ------------------------------------------------------------------
# 5) VISUALIZZA E SALVA (specchio di TUTTI i segmenti, non solo il primo)
#    Nota py2: le keyword devono stare prima di *args (o usare **dict)
# ------------------------------------------------------------------
mirror = multi_segmented_wing.symmetrize()
draw(multi_segmented_wing, *mirror)
save(multi_segmented_wing, fn='trapezoidal_wing.p')
save_IGES(*( [multi_segmented_wing] + mirror ),
          **dict(fn='trapezoidal_wing.igs'))
print('Fatto! %d segmenti fusi. Salvati trapezoidale .p e .igs in play/' % len(ws))
print('La finestra 3D si sblocca quando lo script termina; ESC per chiuderla.')