#!/usr/bin/env python3
import asyncio
import json
import threading

import websockets

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WsToRos2(Node):
    def __init__(self):
        super().__init__("ws_to_ros2_topics")
        self.publishers = {}  # topic_name -> publisher

    def publish_dynamic(self, topic_name: str, value: str):
        """
        Create a publisher for topic_name if it doesn't exist, then publish value.
        """
        # Zorg dat topicnaam ROS2-compatibel is (minimaal: start met '/')
        if not topic_name.startswith("/"):
            ros_topic = "/" + topic_name
        else:
            ros_topic = topic_name

        if ros_topic not in self.publishers:
            pub = self.create_publisher(String, ros_topic, 10)
            self.publishers[ros_topic] = pub
            self.get_logger().info(f"📌 Nieuwe ROS2 publisher aangemaakt: {ros_topic} [std_msgs/String]")

        msg = String()
        msg.data = str(value)
        self.publishers[ros_topic].publish(msg)
        self.get_logger().info(f"📤 Gepubliceerd op {ros_topic}: {msg.data}")


def spin_ros(node: Node):
    """
    Spin ROS in a separate thread so asyncio can run in main thread.
    """
    rclpy.spin(node)


async def websocket_loop(node: WsToRos2, uri: str):
    """
    Connect to WS server, receive JSON, create ROS2 topics, publish values.
    """
    node.get_logger().info(f"🔌 Verbinden met WebSocket server: {uri}")

    async with websockets.connect(uri) as websocket:
        node.get_logger().info("✅ WebSocket verbonden! Wachten op berichten...")

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                node.get_logger().warning(f"❌ Ongeldig JSON ontvangen: {message}")
                continue

            # Verwachte structuur:
            # {
            #   "type": "topic",
            #   "from": "...",
            #   "data": {"name": "topic1", "value": "Any value"}
            # }
            if data.get("type") != "topic":
                node.get_logger().info(f"ℹ️ Bericht genegeerd (type != topic): {data.get('type')}")
                continue

            payload = data.get("data", {})
            topic_name = payload.get("name")
            value = payload.get("value")

            if not topic_name:
                node.get_logger().warning(f"⚠️ 'data.name' ontbreekt in bericht: {data}")
                continue

            # Belangrijk: publish gebeurt in ROS thread-safe context via rclpy
            # Hier is dat ok: rclpy is in andere thread aan het spinnen,
            # publishers zijn thread-safe genoeg voor deze simpele use-case.
            node.publish_dynamic(topic_name, value)


def main():
    uri = "ws://10.147.6.196:9000"

    rclpy.init()
    node = WsToRos2()

    # Start ROS spinning in aparte thread
    ros_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    # Run asyncio WS loop in main thread
    try:
        asyncio.run(websocket_loop(node, uri))
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("🛑 Afsluiten...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
