package ma.enset.nn.factory;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.ReLUActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.activation.TanhActivation;

import java.util.Locale;

/**
 * Factory (amelioration) : centralise la creation des fonctions d'activation.
 */
public final class ActivationFactory {

    private ActivationFactory() {
    }

    public static ActivationFunction create(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Le nom de la fonction d'activation est obligatoire");
        }
        return switch (name.toLowerCase(Locale.ROOT)) {
            case "sigmoid", "sigmoide" -> new SigmoidActivation();
            case "identity", "identite" -> new IdentityActivation();
            case "relu" -> new ReLUActivation();
            case "tanh" -> new TanhActivation();
            case "softmax" -> new SoftmaxActivation();
            default -> throw new IllegalArgumentException("Fonction d'activation inconnue : " + name);
        };
    }
}
