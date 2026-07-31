#!/bin/sh
# Shared fail-closed status protocol for Hermes container initialization.

hermes_current_boot_token() {
    pid_namespace=$(readlink /proc/1/ns/pid 2>/dev/null) || return 1
    pid1_start=$(cut -d ' ' -f 22 /proc/1/stat 2>/dev/null) || return 1
    case "$pid_namespace:$pid1_start" in
        *"
"*|":") return 1 ;;
    esac
    printf '%s:%s' "$pid_namespace" "$pid1_start"
}

hermes_mark_status() {
    marker=$1
    state=$2
    boot_token=$3
    case "$state" in
        failed|ready) ;;
        *) return 2 ;;
    esac
    case "$boot_token" in
        ""|*"
"*) return 2 ;;
    esac

    marker_tmp="${marker}.${state}.$$"
    if ! (umask 077; set -C; printf '%s:%s\n' "$state" "$boot_token" > "$marker_tmp"); then
        rm -f "$marker_tmp"
        return 1
    fi
    if ! chmod 0600 "$marker_tmp" || ! mv -f "$marker_tmp" "$marker"; then
        rm -f "$marker_tmp"
        return 1
    fi
}

hermes_status_is_ready() {
    marker=$1
    boot_token=$2
    status=$(cat "$marker" 2>/dev/null) || return 1
    [ "$status" = "ready:$boot_token" ]
}
