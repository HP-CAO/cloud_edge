import redis


class RedisServerParams:
    def __init__(self):
        self.port = "6379"
        self.password = ""
        self.host_ip = "10.162.12.241"
        self.name = "server_1"


class ChannelMappingParams:
    def __init__(self):
        self.server_name = "server_1"
        self.channel_name = "channel"


class RedisParams:
    def __init__(self):
        self.servers = [RedisServerParams()] * 1
        self.ch_edge_control = ChannelMappingParams()
        self.ch_plant_trajectory_segment = ChannelMappingParams()
        self.ch_edge_weights = ChannelMappingParams()
        self.ch_edge_ready_update = ChannelMappingParams()
        self.ch_plant_reset = ChannelMappingParams()
        self.ch_edge_mode = ChannelMappingParams()
        self.ch_edge_trajectory = ChannelMappingParams()
        self.ch_training_steps = ChannelMappingParams()


class RedisConnection:
    def __init__(self, params: RedisParams):
        self.params = params

        self.pools = {server.name: redis.ConnectionPool(host=server.host_ip, port=server.port,
                                                        password=server.password) for server in self.params.servers}

        self.conns = {name: redis.Redis(connection_pool=pool) for (name, pool) in self.pools.items()}

    def publish(self, channel: ChannelMappingParams, message):
        self.conns[channel.server_name].publish(channel.channel_name, message)

    def subscribe(self, channel: ChannelMappingParams):
        substates = self.conns[channel.server_name].pubsub()
        substates.subscribe(channel.channel_name)
        substates.parse_response()
        return substates
