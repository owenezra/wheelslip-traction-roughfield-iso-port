"""All-zero blind torque program."""


class DoNothingPolicy:
    def act(self, observation):
        _ = observation
        return [0.0] * 14


def load_policy():
    return DoNothingPolicy()
