package ma.enset.nn.observer;

import ma.enset.nn.state.NetworkStateType;

import java.time.Instant;

/**
 * Evenement diffuse aux observateurs lors d'un changement d'etat du reseau.
 */
public record NetworkEvent(String networkName, NetworkStateType state, String message, Instant timestamp) {

    public NetworkEvent {
        if (networkName == null || networkName.isBlank()) {
            throw new IllegalArgumentException("Le nom du reseau est obligatoire");
        }
        if (state == null) {
            throw new IllegalArgumentException("L'etat est obligatoire");
        }
        if (message == null) {
            message = "";
        }
        if (timestamp == null) {
            timestamp = Instant.now();
        }
    }
}
