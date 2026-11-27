package ma.enset.nn.core;

import java.util.List;

/**
 * Contrat commun a toutes les couches. L'interface facilite l'application du
 * pattern Decorator (exigence e) : une couche decoree reste une couche.
 */
public interface Layer {

    double[] forward(double[] inputs);

    void connectTo(Layer nextLayer);

    int neuronCount();

    LayerType type();

    String description();

    List<Neuron> getNeurons();
}
