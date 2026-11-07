"""Bind native SDK PCI observations to the benchmark's actual runtime device."""

from __future__ import annotations

from device_observation import capture_observation


def bind_agent(agents: dict, selected: int, capture: dict, *, rocpd=False) -> dict:
    if (
        not isinstance(agents, dict)
        or type(selected) is not int
        or not 0 < selected < 1 << 64
        or type(rocpd) is not bool
    ):
        raise ValueError("explicit native agent identity and metadata namespace required")
    expected = capture_observation(capture).identity
    if expected.pci_bdf is None:
        raise ValueError("native profiler device binding needs an observed runtime PCI BDF")
    matches = []
    for identifier, row in agents.items():
        # ROCpd extdata is the pinned SDK agent serializer. Its table-level
        # uuid is device_id, not a unique GPU UUID; retain that distinction.
        metadata = row.get("native_agent_metadata") if rocpd else row
        if not isinstance(metadata, dict):
            raise ValueError("native profiler agent metadata is malformed")
        if metadata.get("name") != row.get("name"):
            raise ValueError("native profiler agent metadata architecture conflicts")
        domain, location = metadata.get("domain"), metadata.get("location_id")
        if (
            type(domain) is not int
            or not 0 <= domain <= 0xFFFF
            or type(location) is not int
            or not 0 <= location <= 0xFFFF
        ):
            raise ValueError(
                "native profiler PCI domain/BDF is unavailable or outside the maintained encoding"
            )
        bdf = f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7:x}"
        if bdf == expected.pci_bdf:
            matches.append(identifier)
    if matches != [selected]:
        raise ValueError(
            "native selected agent does not uniquely match the observed execution PCI device"
        )
    return dict(
        state="verified",
        native_agent_id=selected,
        pci_bdf=expected.pci_bdf,
        runtime_ordinal=expected.runtime_ordinal,
        association="native_SDK_domain_and_location_id_to_same_process_runtime_PCI_observation",
        native_uuid_inferred_from_device_id=False,
        ordinal_namespaces_equated=False,
        scope="node_local_observed_PCI_device_not_board_count_or_UUID_equivalence",
    )
