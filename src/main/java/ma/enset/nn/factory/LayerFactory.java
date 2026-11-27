package ma.enset.nn.factory;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.SimpleLayer;

import java.util.Objects;

/**
 * Factory (amelioration) : fabrique de couches pre-configurees.
 */
public final class LayerFactory {

    private LayerFactory() {
    }

    public static Layer createInputLayer(int neuronCount) {
        return new SimpleLayer(LayerType.INPUT, neuronCount, new IdentityActivation());
    }

    public static Layer createHiddenLayer(int neuronCount) {
        return new SimpleLayer(LayerType.HIDDEN, neuronCount, new SigmoidActivation());
    }

    public static Layer createOutputLayer(int neuronCount) {
        return new SimpleLayer(LayerType.OUTPUT, neuronCount, new SoftmaxActivation());
    }

    public static Layer createLayer(LayerType type, int neuronCount, ActivationFunction activation) {
        return new SimpleLayer(Objects.requireNonNull(type, "type"), neuronCount, Objects.requireNonNull(activation, "activation"));
    }
}
