package ma.enset.nn.core;

/**
 * Connexion orientee entre un neurone source et un neurone destination.
 * Le poids W est un reel compris entre 0 et 1, conformement a l'enonce.
 */
public class Connection {

    private final Neuron source;
    private final Neuron destination;
    private double weight;

    public Connection(Neuron source, Neuron destination, double weight) {
        this.source = source;
        this.destination = destination;
        setWeight(weight);
    }

    public Neuron getSource() {
        return source;
    }

    public Neuron getDestination() {
        return destination;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        if (weight < 0.0 || weight > 1.0) {
            throw new IllegalArgumentException("Le poids W doit etre compris entre 0 et 1 : " + weight);
        }
        this.weight = weight;
    }

    @Override
    public String toString() {
        return String.format("Connection[n%s -> n%s, w=%.4f]", source.getId(), destination.getId(), weight);
    }
}
