package ma.enset.nn.activation;

/**
 * Ancienne bibliotheque d'activation : interface differente de celle utilisee
 * par les neurones (Adapter, exigence b).
 */
public interface LegacySquashingFunction {

    double squash(double value);

    String getLabel();
}
