"""All-positive-saturation blind torque program."""


class FullThrottlePolicy:
    def act(self, observation):
        _ = observation
        return [1.0] * 14


def load_policy():
    return FullThrottlePolicy()
