package com.example.javafx;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Group;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.effect.GaussianBlur;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Shape;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.Stage;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.stream.IntStream;

public class HelloFX extends Application {


    @Override
    public void start(Stage primaryStage) throws FileNotFoundException {
        primaryStage.setTitle("Drawing Operations Test");
        Parent root = null;
        try {
            root = FXMLLoader.load(getClass().getResource("hello-view.fxml"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Canvas canvas = new Canvas(800, 800);
        GraphicsContext gc = canvas.getGraphicsContext2D();


        primaryStage.setScene(new Scene(root));
        primaryStage.show();


    }



    public static void main(Stage stage) {
        launch();
    }
}