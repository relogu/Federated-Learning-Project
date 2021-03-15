#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:21:07 2021

@author: relogu
"""
import os
import flask
from flask import request, abort, send_from_directory#, redirect, url_for, jsonify
import simple_fed_avg_only as t1lf
#import midpoint_fed_avg as t1lf
from simple_fed_avg_only import ServerState
import numpy as np
import glob
#from tester_for_decorators import State

UPLOAD_FOLDER = '/home/relogu/OneDrive/UNIBO/Magistrale/GNN_project/py/make_moons_test/SERVER_OUT'
#UPLOAD_FOLDER = '/home/relogu/OneDrive/UNIBO/Magistrale/GNN_project/py/make_moons_test/CLIENTS_OUT'


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def getClientsOutFromFile(n_clients, n_layers):
    """Read the clients' weights from received files."""
    out = [0]*n_clients
    for i in range(n_clients):
        out[i] = readWeightsForClient(i, [0]*n_layers, UPLOAD_FOLDER+"/weights_c=")
    return out

def readWeightsForClient(client, init_weights, filename):
    """Print in two files the weights of the simple keras model."""
    for i in range(len(init_weights)):
        init_weights[i] = np.loadtxt(filename+str(client)+"_"+str(i)+".txt")
    return init_weights

def printWeightsToFile(weights, filename):
    """Print in two files the weights of the simple keras model."""
    i=0
    fn=""
    for w in weights:
        fn=UPLOAD_FOLDER+filename+str(i)+".txt"
        if isinstance(w, type(np.array(0))):
            np.savetxt(fn, w)
        else:
            np.savetxt(fn, [w])
        i+=1

@app.route('/', methods=['GET'])
def home():
    """Respresent the homepage."""
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/api/v1/launch_fed_avg', methods=['GET'])
def launch_fed_avg():
    """Launch the Fed Alg procedure."""
    print("eccolo")

    n_layers = len(glob.glob(UPLOAD_FOLDER+'/ser*'))
    n_clients = len(glob.glob(UPLOAD_FOLDER+'/*'))
    n_clients = (int)(n_clients/n_layers) - 1 
    models = getClientsOutFromFile(n_clients, n_layers)
    states = [.0]*n_clients
    i=0
    for i in range(n_clients):
        print(str(i))
        states[i] = ServerState(model=models[i])
    m=[.0]*n_layers
    for i in range(n_layers):
        m[i] = np.loadtxt(UPLOAD_FOLDER+"/server_"+str(i)+".txt")
    iter_proc = t1lf.build_fed_avg_process(weights=m)
    s = iter_proc.initialize()
    s = iter_proc.next(s, states)
    printWeightsToFile(s.model, "/server_")
    return "", 201

@app.route("/api/v1/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_FOLDER, path, as_attachment=True)

@app.route("/api/v1//files/<filename>", methods=["POST"])
def post_file(filename):
    """Upload a file."""
    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories allowed")

    with open(os.path.join(UPLOAD_FOLDER, filename), "wb") as fp:
        fp.write(request.data)

    # Return 201 CREATED
    return "", 201

app.run()
