package ma.enset.nn.observer;

/**
 * Observer (exigence d) : composant qui souhaite etre notifie des changements
 * d'etat du reseau (par exemple le tableau de bord).
 */
public interface NetworkObserver {

    void onNetworkEvent(NetworkEvent event);
}
