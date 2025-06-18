import src.inno_reservoir.curves.inflow_curves.curve as curve

class IPRCurve(curve.Curve):
    def __init__(self, PI0, pressure0, rmob, sat0, ds) -> None:
        pressure0 = pressure0 * curve.CURVE_UNITS['ipr']['direct']
        PI0 = PI0 * (curve.CURVE_UNITS['ipr']['inverse'] / curve.CURVE_UNITS['ipr']['direct'])
        ds = ds * (1.0 / curve.CURVE_UNITS['ipr']['direct'])

        params = {
            curve.INV_FUNC: lambda bhp: PI0 * (pressure0 - bhp) * (rmob + (1.0 - rmob) * min(max(sat0 + ds * bhp, 0.0), 1.0)),
            curve.FUNC: lambda prod: pressure0 - prod / (PI0 * (rmob + (1.0 - rmob) * min(max(sat0 + ds * pressure0, 0.0), 1.0)))
        }

        super().__init__(params=params, kind='ipr')

class WATCurve(curve.Curve):
    def __init__(self, PI0, pressure0, rmob, sat0, ds) -> None:
        pressure0 = pressure0 * curve.CURVE_UNITS['wat']['inverse']
        PI0 = PI0 * (curve.CURVE_UNITS['wat']['direct'] / curve.CURVE_UNITS['wat']['inverse'])
        ds = ds * (1.0 / curve.CURVE_UNITS['wat']['inverse'])

        params = {
            curve.FUNC: lambda bhp: PI0 * (pressure0 - bhp) * min(max(sat0 + ds * bhp, 0.0), 1.0),
            curve.INV_FUNC: lambda prod: pressure0 - prod / (PI0 * min(max(sat0 + ds * pressure0, 0.0), 1.0))
        }
        super().__init__(params=params, kind='wat')

class OILCurve(curve.Curve):
    def __init__(self, PI0, pressure0, rmob, sat0, ds) -> None:
        pressure0 = pressure0 * curve.CURVE_UNITS['oil']['inverse']
        PI0 = PI0 * (curve.CURVE_UNITS['oil']['direct'] / curve.CURVE_UNITS['oil']['inverse'])
        ds = ds * (1.0 / curve.CURVE_UNITS['oil']['inverse'])

        params = {
            curve.FUNC: lambda bhp: PI0 * (pressure0 - bhp) * rmob * (1.0 - min(max(sat0 + ds * bhp, 0.0), 1.0)),
            curve.INV_FUNC: lambda prod: pressure0 - prod / (PI0 * rmob * (1.0 - min(max(sat0 + ds * pressure0, 0.0), 1.0)))
        }
        super().__init__(params=params, kind='oil')

class GASCurve(curve.Curve):
    def __init__(self, PI0, pressure0, rmob, sat0, ds) -> None:
        pressure0 = pressure0 * curve.CURVE_UNITS['gas']['inverse']
        PI0 = PI0 * (curve.CURVE_UNITS['gas']['direct'] / curve.CURVE_UNITS['gas']['inverse'])
        ds = ds * (1.0 / curve.CURVE_UNITS['gas']['inverse'])

        params = {
            curve.FUNC: lambda bhp: 0.0,
            curve.INV_FUNC: lambda prod: 0.0
        }
        super().__init__(params=params, kind='gas')