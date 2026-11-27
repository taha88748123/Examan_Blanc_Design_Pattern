package ma.enset.nn.core;

import ma.enset.nn.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Neurone : somme les entrees ponderees par les poids des connexions, ajoute le
 * biais b puis applique sa fonction d'activation (Strategy).
 */
public class Neuron {

    private final int id;
    private final ActivationFunction activation;
    private final List<Connection> incomingConnections = new ArrayList<>();
    private double bias;
    private double output;

    public Neuron(int id, ActivationFunction activation) {
        this(id, 0.0, activation);
    }

    public Neuron(int id, double bias, ActivationFunction activation) {
        this.id = id;
        this.bias = bias;
        this.activation = Objects.requireNonNull(activation, "activation");
    }

    public double computeSum() {
        double sum = bias;
        for (Connection connection : incomingConnections) {
            sum += connection.getSource().getOutput() * connection.getWeight();
        }
        return sum;
    }

    public double forward() {
        output = activation.apply(computeSum());
        return output;
    }

    public void addIncomingConnection(Connection connection) {
        incomingConnections.add(Objects.requireNonNull(connection, "connection"));
    }

    public int getId() {
        return id;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public ActivationFunction getActivation() {
        return activation;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public List<Connection> getIncomingConnections() {
        return Collections.unmodifiableList(incomingConnections);
    }
}
