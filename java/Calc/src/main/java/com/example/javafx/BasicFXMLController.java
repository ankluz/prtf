package com.example.javafx;


import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.text.TextAlignment;

import static javafx.scene.text.TextAlignment.*;

public class BasicFXMLController {

    @FXML
    private Label label;
    @FXML
    public void initialize() {
        label.setText("0");

    }
    boolean first = false;
    private String Text;
    @FXML
    private void handleButtonAction(ActionEvent event) {

        var a = event.getTarget().toString().split("");
        if (first==false){
            label.setText(a[a.length-2]);
            Text = a[a.length-2];
            first=true;
        } else {

            Text+=a[a.length-2];
            label.setText(Text);
        }


    }
    boolean minus_flag = false;
    @FXML
    private void handleplusminus(ActionEvent event) {
        if (minus_flag==false){
            minus_flag=true;
        } else {
            minus_flag=false;
        }
        var a = event.getTarget().toString().split("");

        label.setText(a[a.length-2]);
    }

}