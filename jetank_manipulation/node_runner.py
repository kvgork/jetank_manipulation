#!/usr/bin/env python3
"""
Shared entry-point runner for the jetank_manipulation executables.

Every node in this package uses the same lifecycle: ``rclpy.init`` -> construct
the node -> spin it on a ``MultiThreadedExecutor`` (the package-wide idiom —
service/action callbacks run in reentrant groups and block on futures serviced
by sibling executor threads) -> destroy the node -> guarded ``rclpy.shutdown``.

``spin_node`` centralises that boilerplate so each node's ``main()`` is a
one-liner and the CMake ``RENAME`` executables stay unchanged.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor


def spin_node(node_factory, args=None) -> None:
    """
    Init rclpy, spin ``node_factory()`` on a MultiThreadedExecutor, clean up.

    ``node_factory`` is typically the Node subclass itself. KeyboardInterrupt
    exits quietly; shutdown is guarded because the context may already have
    been torn down externally (e.g. by a launch-driven SIGINT).
    """
    rclpy.init(args=args)
    node = node_factory()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
