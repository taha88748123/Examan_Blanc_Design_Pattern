package ma.enset.nn;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.ReLUActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.activation.TanhActivation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ActivationFunctionTest {

    @Test
    void sigmoidShouldBeBoundedAndSymmetric() {
        ActivationFunction sigmoid = new SigmoidActivation();
        assertEquals(0.5, sigmoid.apply(0.0), 1e-9);
        assertTrue(sigmoid.apply(5.0) > 0.99);
        assertTrue(sigmoid.apply(-5.0) < 0.01);
        assertEquals(1.0, sigmoid.apply(2.0) + sigmoid.apply(-2.0), 1e-9);
        assertEquals("Sigmoid", sigmoid.name());
    }

    @Test
    void identityShouldReturnInput() {
        ActivationFunction identity = new IdentityActivation();
        assertEquals(-2.5, identity.apply(-2.5), 0.0);
        assertEquals("Identity", identity.name());
    }

    @Test
    void reluShouldClipNegativeValues() {
        ActivationFunction relu = new ReLUActivation();
        assertEquals(0.0, relu.apply(-3.0), 0.0);
        assertEquals(2.5, relu.apply(2.5), 0.0);
        assertEquals("ReLU", relu.name());
    }

    @Test
    void tanhShouldStayInMinusOneOne() {
        ActivationFunction tanh = new TanhActivation();
        assertEquals(0.0, tanh.apply(0.0), 1e-9);
        assertEquals(1.0, tanh.apply(10.0), 1e-6);
        assertEquals(-1.0, tanh.apply(-10.0), 1e-6);
    }

    @Test
    void softmaxShouldReturnProbabilityDistribution() {
        ActivationFunction softmax = new SoftmaxActivation();
        double[] outputs = softmax.apply(new double[]{1.0, 2.0, 3.0});

        assertEquals(3, outputs.length);
        assertEquals(1.0, outputs[0] + outputs[1] + outputs[2], 1e-9);
        assertTrue(outputs[2] > outputs[1]);
        assertTrue(outputs[1] > outputs[0]);
        assertTrue(softmax.isVectorBased());
    }
}
