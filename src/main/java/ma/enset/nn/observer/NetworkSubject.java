package ma.enset.nn.observer;

/**
 * Sujet observable : contrat de gestion des observateurs.
 */
public interface NetworkSubject {

    void addObserver(NetworkObserver observer);

    void removeObserver(NetworkObserver observer);

    void notifyObservers(NetworkEvent event);
}
