from abc import abstractmethod

from utils import setup_model


class ExperimentHelper:
    """Essentially the run script will call this. Then will do experiment.run_experiment()"""

    def __init__(self, config):
        self.config = config

        # perform checks
        self.check_config()

        # get important stuff
        self.seed = self.config["seed"]

    def check_config(self):
        """Add checks you think are important here"""
        assert "seed" in self.config, "Ensure seed present in config"
        assert "output_dir" in self.config, "Ensure output_dir present in config"

    @abstractmethod
    def read_data(self):
        """Reads data"""
        pass

    @abstractmethod
    def perform_preprocessing(self):
        """Perform preprocessing"""
        pass

    def setup_model(self):
        """Setup model"""
        return setup_model(self.config)

    @abstractmethod
    def fit_model(self, x, y, model):
        pass

    @abstractmethod
    def run_experiment(self):
        """Runs experiment, kind of the only method to be called in the main script"""
        pass
