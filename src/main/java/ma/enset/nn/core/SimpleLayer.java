package ma.enset.nn.core;

import ma.enset.nn.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Couche concrete d'un reseau entierement connecte.
 */
public class SimpleLayer implements Layer {

    private final LayerType type;
    private final List<Neuron> neurons = new ArrayList<>();
    private final Random random;

    public SimpleLayer(LayerType type, int neuronCount, ActivationFunction activation) {
        this(type, neuronCount, activation, new Random());
    }

    public SimpleLayer(LayerType type, int neuronCount, ActivationFunction activation, Random random) {
        if (neuronCount <= 0) {
            throw new IllegalArgumentException("Une couche doit contenir au moins un neurone");
        }
        this.type = Objects.requireNonNull(type, "type");
        this.random = Objects.requireNonNull(random, "random");
        for (int i = 0; i < neuronCount; i++) {
            neurons.add(new Neuron(i, activation));
        }
    }

    @Override
    public double[] forward(double[] inputs) {
        Objects.requireNonNull(inputs, "inputs");

        if (type == LayerType.INPUT) {
            if (inputs.length != neurons.size()) {
                throw new IllegalArgumentException(
                        "Taille d'entree invalide : attendu " + neurons.size() + ", recu " + inputs.length);
            }
            for (int i = 0; i < inputs.length; i++) {
                neurons.get(i).setOutput(inputs[i]);
            }
            return inputs.clone();
        }

        int expectedInputs = neurons.get(0).getIncomingConnections().size();
        if (expectedInputs > 0 && inputs.length != expectedInputs) {
            throw new IllegalArgumentException(
                    "Taille d'entree invalide : attendu " + expectedInputs + ", recu " + inputs.length);
        }

        double[] sums = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            sums[i] = neurons.get(i).computeSum();
        }

        ActivationFunction activation = neurons.get(0).getActivation();
        double[] outputs;
        if (activation.isVectorBased()) {
            outputs = activation.apply(sums);
        } else {
            outputs = new double[sums.length];
            for (int i = 0; i < sums.length; i++) {
                outputs[i] = neurons.get(i).getActivation().apply(sums[i]);
            }
        }

        for (int i = 0; i < outputs.length; i++) {
            neurons.get(i).setOutput(outputs[i]);
        }
        return outputs;
    }

    @Override
    public void connectTo(Layer nextLayer) {
        Objects.requireNonNull(nextLayer, "nextLayer");
        for (Neuron target : nextLayer.getNeurons()) {
            for (Neuron source : neurons) {
                target.addIncomingConnection(new Connection(source, target, random.nextDouble()));
            }
        }
    }

    @Override
    public int neuronCount() {
        return neurons.size();
    }

    @Override
    public LayerType type() {
        return type;
    }

    @Override
    public String description() {
        String label = switch (type) {
            case INPUT -> "InputLayer";
            case HIDDEN -> "HiddenLayer";
            case OUTPUT -> "OutputLayer";
        };
        return String.format("%s[%d] (%s)", label, neurons.size(), neurons.get(0).getActivation().name());
    }

    @Override
    public List<Neuron> getNeurons() {
        return Collections.unmodifiableList(neurons);
    }
}
