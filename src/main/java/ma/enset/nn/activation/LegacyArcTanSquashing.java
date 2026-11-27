package ma.enset.nn.activation;

/**
 * Ancienne fonction d'activation (arc tangente normalisee) qui ne connait pas
 * l'interface ActivationFunction.
 */
public class LegacyArcTanSquashing implements LegacySquashingFunction {

    @Override
    public double squash(double value) {
        return Math.atan(value) * 2.0 / Math.PI;
    }

    @Override
    public String getLabel() {
        return "LegacyArcTan";
    }
}
