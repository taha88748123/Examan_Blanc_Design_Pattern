package ma.enset.nn;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.ReLUActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.activation.TanhActivation;
import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.factory.ActivationFactory;
import ma.enset.nn.factory.LayerFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FactoryTest {

    @Test
    void activationFactoryShouldCreateEachKnownFunction() {
        assertInstanceOf(SigmoidActivation.class, ActivationFactory.create("sigmoid"));
        assertInstanceOf(IdentityActivation.class, ActivationFactory.create("identity"));
        assertInstanceOf(ReLUActivation.class, ActivationFactory.create("relu"));
        assertInstanceOf(TanhActivation.class, ActivationFactory.create("tanh"));
        assertInstanceOf(SoftmaxActivation.class, ActivationFactory.create("softmax"));
    }

    @Test
    void activationFactoryShouldBeCaseInsensitiveAndSupportFrenchAliases() {
        assertInstanceOf(SigmoidActivation.class, ActivationFactory.create("SIGMOIDE"));
        assertEquals("Softmax", ActivationFactory.create("SoftMax").name());
        assertEquals("Identity", ActivationFactory.create("Identite").name());
    }

    @Test
    void activationFactoryShouldRejectUnknownFunction() {
        assertThrows(IllegalArgumentException.class, () -> ActivationFactory.create("gaussian"));
        assertThrows(IllegalArgumentException.class, () -> ActivationFactory.create(""));
    }

    @Test
    void layerFactoryShouldCreatePreconfiguredLayers() {
        Layer input = LayerFactory.createInputLayer(3);
        Layer hidden = LayerFactory.createHiddenLayer(5);
        Layer output = LayerFactory.createOutputLayer(2);

        assertEquals(LayerType.INPUT, input.type());
        assertEquals(LayerType.HIDDEN, hidden.type());
        assertEquals(LayerType.OUTPUT, output.type());
        assertEquals(3, input.neuronCount());
        assertEquals(5, hidden.neuronCount());
        assertEquals(2, output.neuronCount());
        assertEquals("Identity", input.getNeurons().get(0).getActivation().name());
        assertEquals("Sigmoid", hidden.getNeurons().get(0).getActivation().name());
        assertEquals("Softmax", output.getNeurons().get(0).getActivation().name());

        ActivationFunction relu = ActivationFactory.create("relu");
        Layer custom = LayerFactory.createLayer(LayerType.HIDDEN, 4, relu);
        assertEquals(4, custom.neuronCount());
        assertEquals("ReLU", custom.getNeurons().get(0).getActivation().name());
    }
}
