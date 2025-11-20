"""Model architectures for HeartBeat AI."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    MaxPooling1D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


@dataclass
class HeartbeatEnsemble:
    """Dual-branch ensemble combining waveform + spectrogram encoders."""

    num_classes: int = 6
    classes: Sequence[str] = (
        "normal_heart",
        "murmur",
        "extrasystole",
        "normal_resp",
        "wheeze",
        "crackle",
    )
    waveform_input_shape: Tuple[int, int] = (20000, 1)
    spectrogram_input_shape: Tuple[int, int, int] = (128, 156, 1)
    learning_rate: float = 1e-3
    kernel_regularizer: float = 1e-4
    waveform_model: Optional[Model] = field(default=None, init=False)
    spectrogram_model: Optional[Model] = field(default=None, init=False)
    ensemble_model: Optional[Model] = field(default=None, init=False)

    def build_waveform_branch(self) -> Model:
        """1D CNN encoder for raw waveform segments."""
        inputs = Input(shape=self.waveform_input_shape, name="waveform_input")
        x = inputs
        for filters, kernel, pool in [(32, 11, 2), (64, 9, 2), (128, 7, 2), (256, 5, 2)]:
            x = Conv1D(
                filters,
                kernel_size=kernel,
                padding="same",
                kernel_regularizer=l2(self.kernel_regularizer),
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPooling1D(pool_size=pool)(x)
            x = Dropout(0.2)(x)
        x = GlobalAveragePooling1D()(x)
        self.waveform_model = Model(inputs=inputs, outputs=x, name="waveform_branch")
        return self.waveform_model

    def build_spectrogram_branch(self) -> Model:
        """2D CNN encoder for log-mel spectrograms."""
        inputs = Input(shape=self.spectrogram_input_shape, name="spectrogram_input")
        x = inputs
        for idx, filters in enumerate((32, 64, 128, 256)):
            x = Conv2D(
                filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=l2(self.kernel_regularizer),
                name=f"spec_conv_{idx}",
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
        x = GlobalAveragePooling2D()(x)
        self.spectrogram_model = Model(
            inputs=inputs,
            outputs=x,
            name="spectrogram_branch",
        )
        return self.spectrogram_model

    def build_ensemble(self) -> Model:
        """Fuse both encoders and compile classification head."""
        waveform_model = self.waveform_model or self.build_waveform_branch()
        spectrogram_model = self.spectrogram_model or self.build_spectrogram_branch()

        combined = Concatenate(name="fusion_concat")(
            [waveform_model.output, spectrogram_model.output]
        )
        x = Dense(256, activation="relu")(combined)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation="softmax", name="predictions")(x)

        self.ensemble_model = Model(
            inputs=[waveform_model.input, spectrogram_model.input],
            outputs=outputs,
            name="heartbeat_ensemble",
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        self.ensemble_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.ensemble_model

    def summary(self) -> str:
        """Return a textual summary of the ensemble."""
        if not self.ensemble_model:
            return "Model not built yet."
        buffer = io.StringIO()
        self.ensemble_model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
        return buffer.getvalue()
