package ma.enset.nn.aspect;

import java.util.Objects;

/**
 * Contexte de securite simplifie : role de l'utilisateur courant.
 */
public final class SecurityContext {

    public static final String ADMIN_ROLE = "ADMIN";
    public static final String DATA_SCIENTIST_ROLE = "DATA_SCIENTIST";
    public static final String GUEST_ROLE = "GUEST";

    private static final ThreadLocal<String> CURRENT_ROLE = ThreadLocal.withInitial(() -> DATA_SCIENTIST_ROLE);

    private SecurityContext() {
    }

    public static String getRole() {
        return CURRENT_ROLE.get();
    }

    public static void setRole(String role) {
        CURRENT_ROLE.set(Objects.requireNonNull(role, "role"));
    }

    public static void clear() {
        CURRENT_ROLE.remove();
    }

    public static boolean hasRole(String requiredRole) {
        if (requiredRole == null || requiredRole.isBlank()) {
            return true;
        }
        String role = CURRENT_ROLE.get();
        return ADMIN_ROLE.equalsIgnoreCase(role) || requiredRole.equalsIgnoreCase(role);
    }
}
