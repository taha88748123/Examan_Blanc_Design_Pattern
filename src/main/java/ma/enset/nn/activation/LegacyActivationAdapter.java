package ma.enset.nn.activation;

import java.util.Objects;

/**
 * Adapter (exigence b) : adapte une LegacySquashingFunction a l'interface
 * ActivationFunction utilisee par les neurones, sans modifier l'ancien code.
 */
public class LegacyActivationAdapter implements ActivationFunction {

    private final LegacySquashingFunction legacyFunction;

    public LegacyActivationAdapter(LegacySquashingFunction legacyFunction) {
        this.legacyFunction = Objects.requireNonNull(legacyFunction, "legacyFunction");
    }

    @Override
    public double apply(double input) {
        return legacyFunction.squash(input);
    }

    @Override
    public String name() {
        return "Adapter[" + legacyFunction.getLabel() + "]";
    }

    public LegacySquashingFunction getLegacyFunction() {
        return legacyFunction;
    }
}
