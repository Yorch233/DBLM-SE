"""
Author: Yorch233
GitHub: https://github.com/Yorch233/DBLM-SE  
Email: qyao@stmail.ujs.edu.cn
Date: 2025-08-05
"""

import warnings
from typing import Any, Callable, Dict, List, Union


class Register(dict):
    """
    A custom dictionary subclass designed to register and manage artifacts.
    
    Artifacts can be registered by name, allowing for easy retrieval and management
    of components such as models, datasets, or processors. Uses decorator pattern
    for convenient registration.
    
    Example:
        register = Register()
        
        @register.register('my_model')
        class MyModel:
            pass
            
        model_class = register.fetch('my_model')
    """

    def __init__(self, types_list: List[str] = None) -> None:
        """
        Initializes the Register object. Optionally initializes with a list of predefined names.

        Parameters:
            types_list (List[str], optional): A list of artifact names to pre-initialize 
                                            the register with empty dictionaries. 
                                            Defaults to None.
        """
        super().__init__()
        self._dict: Dict[str, Dict[str, Any]] = {}

        # Pre-initialize register with specified artifact names if provided
        if types_list is not None:
            for artifact_type in types_list:
                self._dict[artifact_type] = {}

    def register(self, artifact_name: str) -> Callable[[Any], Any]:
        """
        Decorator method to register an artifact by its name.
        
        This method returns a decorator that can be used to register classes,
        functions, or any other objects with a string identifier for later retrieval.

        Parameters:
            artifact_name (str): The unique name/identifier for the artifact to be registered.

        Returns:
            Callable[[Any], Any]: A decorator function that registers the decorated object.
            
        Example:
            register = Register()
            
            @register.register('audio_encoder')
            class AudioEncoder:
                pass
        """

        def decorator(artifact: Any) -> Any:
            """
            Inner decorator function that performs the actual registration.
            
            Parameters:
                artifact (Any): The object to be registered (class, function, etc.).
                
            Returns:
                Any: The same artifact object, unchanged.
            """
            self._dict[artifact_name] = artifact
            return artifact

        return decorator

    def fetch(self, name_or_name_list: Union[str, List[str]]) -> Any:
        """
        Retrieves one or more registered artifacts by their names.
        
        Supports both single artifact retrieval and batch retrieval of multiple artifacts.
        For batch retrieval, returns a dictionary mapping names to artifacts, with warnings
        for any unregistered names.

        Parameters:
            name_or_name_list (Union[str, List[str]]): Either a single artifact name 
                                                     or a list of artifact names to retrieve.

        Returns:
            Any: For single retrieval: the registered artifact.
                 For batch retrieval: a dictionary of registered artifacts.
                 
        Raises:
            KeyError: If a single requested artifact name is not registered.
            ValueError: If batch retrieval results in no valid artifacts found.
            
        Example:
            # Single retrieval
            model = register.fetch('audio_encoder')
            
            # Batch retrieval
            artifacts = register.fetch(['audio_encoder', 'audio_decoder'])
        """
        # Handle batch retrieval of multiple artifacts
        if isinstance(name_or_name_list, list):
            artifacts = {}
            for name in name_or_name_list:
                if name in self._dict:
                    artifacts[name] = self._dict[name]
                else:
                    # Warn about unregistered artifacts but continue processing
                    warnings.warn(
                        f"Unregistered artifact_name: '{name}'. Ignoring this artifact."
                    )
            # Raise error if no valid artifacts were found
            if len(artifacts) == 0:
                raise ValueError("No registered artifacts found.")
            return artifacts
            
        # Handle single artifact retrieval
        elif isinstance(name_or_name_list, str):
            if name_or_name_list in self._dict:
                return self._dict[name_or_name_list]
            else:
                # Raise error for unregistered single artifact
                raise KeyError(
                    f"Unregistered artifact_name: '{name_or_name_list}'.")